require 'nn'
require 'string'
require 'nngraph'

require 's2sa.models'
require 's2sa.data'

local path = require 'pl.path'
local stringx = require 'pl.stringx'
local utf8 --loaded when using character models only

if type(string.split) ~= "function" then
  require 's2sa.string_utils'
end

-- globals
local PAD = 1
local UNK = 2
local START = 3
local END = 4
local UNK_WORD = '<unk>'
local START_WORD = '<s>'
local END_WORD = '</s>'
local START_CHAR = '{'
local END_CHAR = '}'
local State
local model
local model_opt
local idx2feature_src = {}
local feature2idx_src = {}
local idx2feature_targ = {}
local feature2idx_targ = {}
local word2charidx_targ
local init_fwd_enc = {}
local init_fwd_dec = {}
local idx2word_src
local word2idx_src
local idx2word_targ
local word2idx_targ
local context_proto
local context_proto2
local decoder_softmax
local decoder_attn
local phrase_table
local word_vecs_enc
local word_vecs_dec
local softmax_layers
local hop_attn
local char2idx
local idx2char
local info

local opt = {}
local cmd = torch.CmdLine()

-- file location
cmd:option('-model', 'seq2seq_lstm_attn.t7.', [[Path to model .t7 file]])
cmd:option('-src_file', '', [[Source sequence to decode (one line per sequence)]])
cmd:option('-targ_file', '', [[True target sequence (optional)]])
cmd:option('-output_file', 'pred.txt', [[Path to output the predictions (each line will be the decoded sequence]])
cmd:option('-src_dict', '', [[Path to source vocabulary (*.src.dict file)]])
cmd:option('-targ_dict', '', [[Path to target vocabulary (*.targ.dict file)]])
cmd:option('-feature_dict_prefix', '', [[Prefix of the path to features vocabularies (*.feature_N.dict file)]])
cmd:option('-char_dict', '', [[If using chars, path to character vocabulary (*.char.dict file)]])

-- beam search options
cmd:option('-beam', 5,[[Beam size]])
cmd:option('-max_sent_l', 250, [[Maximum sentence length. If any sequences in srcfile are longer than this then it will error out]])
cmd:option('-simple', 0, [[If = 1, output prediction is simply the first time the top of the beam
                         ends with an end-of-sentence token. If = 0, the model considers all
                         hypotheses that have been generated so far that ends with end-of-sentence
                         token and takes the highest scoring of all of them.]])
cmd:option('-replace_unk', 0, [[Replace the generated UNK tokens with the source token that
                              had the highest attention weight. If srctarg_dict is provided,
                              it will lookup the identified source token and give the corresponding
                              target token. If it is not provided (or the identified source token
                              does not exist in the table) then it will copy the source token]])
cmd:option('-srctarg_dict', 'data/en-de.dict', [[Path to source-target dictionary to replace UNK
                                               tokens. See README.md for the format this file should be in]])
cmd:option('-score_gold', 1, [[If = 1, score the log likelihood of the gold as well]])
cmd:option('-n_best', 1, [[If > 1, it will also output an n_best list of decoded sentences]])
cmd:option('-gpuid', -1, [[ID of the GPU to use (-1 = use CPU)]])
cmd:option('-gpuid2', -1, [[Second GPU ID]])
cmd:option('-cudnn', 0, [[If using character model, this should be = 1 if the character model was trained using cudnn]])

local function copy(orig)
  local orig_type = type(orig)
  local copy_obj
  if orig_type == 'table' then
    copy_obj = {}
    for orig_key, orig_value in pairs(orig) do
      copy_obj[orig_key] = orig_value
    end
  else
    copy_obj = orig
  end
  return copy_obj
end

local function append_table(dst, src)
  for i = 1, #src do
    table.insert(dst, src[i])
  end
end

local StateAll = torch.class("StateAll")

function StateAll.initial(start)
  return {start}
end

function StateAll.advance(state, token)
  local new_state = copy(state)
  table.insert(new_state, token)
  return new_state
end

function StateAll.disallow(out)
  local bad = {1, 3} -- 1 is PAD, 3 is BOS
  for j = 1, #bad do
    out[bad[j]] = -1e9
  end
end

function StateAll.same(state1, state2)
  for i = 2, #state1 do
    if state1[i] ~= state2[i] then
      return false
    end
  end
  return true
end

function StateAll.next(state)
  return state[#state]
end

function StateAll.heuristic()
  return 0
end

function StateAll.print(state)
  for i = 1, #state do
    io.write(state[i] .. " ")
  end
  print()
end

-- Convert a flat index to a row-column tuple.
local function flat_to_rc(v, flat_index)
  local row = math.floor((flat_index - 1) / v:size(2)) + 1
  return row, (flat_index - 1) % v:size(2) + 1
end

local function generate_beam(K, max_sent_l, source, source_features, gold, gold_features)

  --reset decoder initial states
  local initial = State.initial(START)

  if opt.gpuid >= 0 and opt.gpuid2 >= 0 then
    cutorch.setDevice(opt.gpuid)
  end
  local n = max_sent_l
  -- Backpointer table.
  local prev_ks = torch.LongTensor(n, K):fill(1)
  -- Current States.
  local next_ys = torch.LongTensor(n, K):fill(1)
  local next_ys_features = {}
  if model_opt.num_target_features > 0 then
    for i = 1, n do
      table.insert(next_ys_features, {})
      for j = 1, model_opt.num_target_features do
        local t
        if model_opt.target_features_lookup[j] == true then
          t = torch.DoubleTensor(K):fill(1)
        else
          t = torch.DoubleTensor(K, #idx2feature_targ[j]):zero()
        end
        table.insert(next_ys_features[i], t)
      end
    end
  end

  -- Current Scores.
  local scores = torch.FloatTensor(n, K)
  scores:zero()
  local source_l = math.min(source:size(1), opt.max_sent_l)
  local attn_argmax = {{initial}} -- store attn weights
  local states = {{initial}} -- store predicted word idx

  next_ys[1][1] = State.next(initial)
  for j = 1, model_opt.num_target_features do
    if model_opt.target_features_lookup[j] == true then
      next_ys_features[1][j][1] = UNK
    else
      next_ys_features[1][j][1][UNK] = 1
    end
  end

  local source_input
  if model_opt.use_chars_enc == 1 then
    source_input = source:view(source_l, 1, source:size(2)):contiguous()
  else
    source_input = source:view(source_l, 1)
  end

  local rnn_state_enc = {}
  for i = 1, #init_fwd_enc do
    table.insert(rnn_state_enc, init_fwd_enc[i]:zero())
  end
  local context = context_proto[{{}, {1,source_l}}]:clone() -- 1 x source_l x rnn_size

  for t = 1, source_l do
    local encoder_input = {source_input[t]}
    if model_opt.num_source_features > 0 then
      append_table(encoder_input, source_features[t])
    end
    append_table(encoder_input, rnn_state_enc)
    local out = model[1]:forward(encoder_input)
    rnn_state_enc = out
    context[{{},t}]:copy(out[#out])
  end
  local rnn_state_dec = {}
  for i = 1, #init_fwd_dec do
    table.insert(rnn_state_dec, init_fwd_dec[i]:zero())
  end

  if model_opt.init_dec == 1 then
    for L = 1, model_opt.num_layers do
      rnn_state_dec[L*2-1+model_opt.input_feed]:copy(
        rnn_state_enc[L*2-1]:expand(K, model_opt.rnn_size))
      rnn_state_dec[L*2+model_opt.input_feed]:copy(
        rnn_state_enc[L*2]:expand(K, model_opt.rnn_size))
    end
  end

  if model_opt.brnn == 1 then
    for i = 1, #rnn_state_enc do
      rnn_state_enc[i]:zero()
    end
    for t = source_l, 1, -1 do
      local encoder_input = {source_input[t]}
      if model_opt.num_source_features > 0 then
        append_table(encoder_input, source_features[t])
      end
      append_table(encoder_input, rnn_state_enc)
      local out = model[4]:forward(encoder_input)
      rnn_state_enc = out
      context[{{},t}]:add(out[#out])
    end
    if model_opt.init_dec == 1 then
      for L = 1, model_opt.num_layers do
        rnn_state_dec[L*2-1+model_opt.input_feed]:add(
          rnn_state_enc[L*2-1]:expand(K, model_opt.rnn_size))
        rnn_state_dec[L*2+model_opt.input_feed]:add(
          rnn_state_enc[L*2]:expand(K, model_opt.rnn_size))
      end
    end
  end

  local rnn_state_dec_gold
  if opt.score_gold == 1 and gold ~= nil then
    rnn_state_dec_gold = {}
    for i = 1, #rnn_state_dec do
      table.insert(rnn_state_dec_gold, rnn_state_dec[i][{{1}}]:clone())
    end
  end

  context = context:expand(K, source_l, model_opt.rnn_size)

  if opt.gpuid >= 0 and opt.gpuid2 >= 0 then
    cutorch.setDevice(opt.gpuid2)
    local context2 = context_proto2[{{1, K}, {1, source_l}}]
    context2:copy(context)
    context = context2
  end

  local out_float = torch.FloatTensor()

  local i = 1
  local done = false
  local max_score = -1e9
  local found_eos = false
  local end_attn_argmax
  local end_hyp
  local end_score
  local max_hyp
  local max_attn_argmax

  local feats_hyp = {}
  for k = 1, K do
    table.insert(feats_hyp, {})
  end

  while (not done) and (i < n) do
    i = i+1
    states[i] = {}
    attn_argmax[i] = {}
    local decoder_input1
    local decoder_input1_features
    if model_opt.use_chars_dec == 1 then
      decoder_input1 = word2charidx_targ:index(1, next_ys:narrow(1,i-1,1):squeeze())
    else
      decoder_input1 = next_ys:narrow(1,i-1,1):squeeze()
      if opt.beam == 1 then
        decoder_input1 = torch.LongTensor({decoder_input1})
      end
      decoder_input1_features = {}
      for j = 1, model_opt.num_target_features do
        table.insert(decoder_input1_features, next_ys_features[i-1][j])
      end
    end
    local decoder_input = {decoder_input1}
    if model_opt.num_target_features > 0 then
      append_table(decoder_input, decoder_input1_features)
    end
    if model_opt.attn == 1 then
      append_table(decoder_input, {context})
    else
      append_table(decoder_input, {context[{{}, source_l}]})
    end
    append_table(decoder_input, rnn_state_dec)
    local out_decoder = model[2]:forward(decoder_input)
    local out_decoder_pred_idx = #out_decoder
    if model_opt.guided_alignment == 1 then
      out_decoder_pred_idx = #out_decoder-1
    end
    local out = model[3]:forward(out_decoder[out_decoder_pred_idx]) -- K x vocab_size

    rnn_state_dec = {} -- to be modified later
    if model_opt.input_feed == 1 then
      table.insert(rnn_state_dec, out_decoder[out_decoder_pred_idx])
    end
    for j = 1, out_decoder_pred_idx - 1 do
      table.insert(rnn_state_dec, out_decoder[j])
    end
    if type(out) == "table" then
      out_float:resize(out[1]:size()):copy(out[1])
    else
      out_float:resize(out:size()):copy(out)
    end
    for k = 1, K do
      State.disallow(out_float:select(1, k))
      out_float[k]:add(scores[i-1][k])
    end
    -- All the scores available.

    local flat_out = out_float:view(-1)
    if i == 2 then
      flat_out = out_float[1] -- all outputs same for first batch
    end

    if model_opt.start_symbol == 1 then
      decoder_softmax.output[{{},1}]:zero()
      decoder_softmax.output[{{},source_l}]:zero()
    end

    for k = 1, K do
      while true do
        local score, index = flat_out:max(1)
        score = score[1]
        local prev_k, y_i = flat_to_rc(out_float, index[1])
        states[i][k] = State.advance(states[i-1][prev_k], y_i)
        local diff = true
        for k2 = 1, k-1 do
          if State.same(states[i][k2], states[i][k]) then
            diff = false
          end
        end

        if i < 2 or diff then
          if model_opt.attn == 1 then
            attn_argmax[i][k] = State.advance(attn_argmax[i-1][prev_k], decoder_softmax.output[prev_k]:clone())
          end
          prev_ks[i][k] = prev_k
          next_ys[i][k] = y_i
          scores[i][k] = score
          flat_out[index[1]] = -1e9
          break -- move on to next k
        end
        flat_out[index[1]] = -1e9
      end
    end
    for j = 1, #rnn_state_dec do
      rnn_state_dec[j]:copy(rnn_state_dec[j]:index(1, prev_ks[i]))
    end

    if model_opt.num_target_features > 0 then
      for k = 1, K do
        table.insert(feats_hyp[k], {})
        for j = 1, model_opt.num_target_features do
          local lk, idx = torch.sort(out[1+j][k], true)
          local best = 1
          local hyp = {}
          if model_opt.target_features_lookup[j] == true then
            next_ys_features[i][j][k] = idx[best]
            hyp[1] = idx[best]
          else
            next_ys_features[i][j]:copy(out[1+j])
            table.insert(hyp, idx[best])
            for l = best+1, lk:size(1) do
              if lk[best] - lk[l] < 0.05 then
                if idx[l] > END then
                  table.insert(hyp, idx[l])
                end
              else
                break
              end
            end
          end
          table.insert(feats_hyp[k][i-1], hyp)
        end
      end
    end

    end_hyp = states[i][1]
    end_score = scores[i][1]
    if model_opt.attn == 1 then
      end_attn_argmax = attn_argmax[i][1]
    end
    if end_hyp[#end_hyp] == END then
      done = true
      found_eos = true
    else
      for k = 1, K do
        local possible_hyp = states[i][k]
        if possible_hyp[#possible_hyp] == END then
          found_eos = true
          if scores[i][k] > max_score then
            max_hyp = possible_hyp
            max_score = scores[i][k]
            if model_opt.attn == 1 then
              max_attn_argmax = attn_argmax[i][k]
            end
          end
        end
      end
    end
  end
  local gold_score = 0
  if opt.score_gold == 1 and gold ~= nil then
    rnn_state_dec = {}
    for fwd_i = 1, #init_fwd_dec do
      table.insert(rnn_state_dec, init_fwd_dec[fwd_i][{{1}}]:zero())
    end
    if model_opt.init_dec == 1 then
      rnn_state_dec = rnn_state_dec_gold
    end
    local target_l = gold:size(1)
    for t = 2, target_l do
      local decoder_input1
      if model_opt.use_chars_dec == 1 then
        decoder_input1 = word2charidx_targ:index(1, gold[{{t-1}}])
      else
        decoder_input1 = gold[{{t-1}}]
      end
      local decoder_input = {decoder_input1}
      if model_opt.num_target_features > 0 then
        append_table(decoder_input, gold_features[t-1])
      end
      if model_opt.attn == 1 then
        append_table(decoder_input, {context[{{1}}]})
      else
        append_table(decoder_input, {context[{{1}, source_l}]})
      end
      append_table(decoder_input, rnn_state_dec)
      local out_decoder = model[2]:forward(decoder_input)
      local out = model[3]:forward(out_decoder[#out_decoder]) -- K x vocab_size
      rnn_state_dec = {} -- to be modified later
      if model_opt.input_feed == 1 then
        table.insert(rnn_state_dec, out_decoder[#out_decoder])
      end
      for j = 1, #out_decoder - 1 do
        table.insert(rnn_state_dec, out_decoder[j])
      end
      if type(out) == "table" then
        gold_score = gold_score + out[1][1][gold[t]]
      else
        gold_score = gold_score + out[1][gold[t]]
      end
    end
  end
  if opt.simple == 1 or end_score > max_score or not found_eos then
    max_hyp = end_hyp
    max_score = end_score
    max_attn_argmax = end_attn_argmax
  end

  local max_feats_hyp = {}

  if model_opt.num_target_features > 0 then
    for j = 1, i-1 do
      table.insert(max_feats_hyp, {})
    end

    -- follow beam path to build the features sequence
    local k = 1
    for j = i, 2, -1 do
      k = prev_ks[j][k]
      max_feats_hyp[j-1] = feats_hyp[k][j-1]
    end
  end

  return max_hyp, max_feats_hyp, max_score, max_attn_argmax, gold_score, states[i], scores[i], attn_argmax[i]
end

local function idx2key(file)
  local f = io.open(file,'r')
  local t = {}
  for line in f:lines() do
    local c = {}
    for w in line:gmatch'([^%s]+)' do
      table.insert(c, w)
    end
    t[tonumber(c[2])] = c[1]
  end
  return t
end

local function flip_table(u)
  local t = {}
  for key, value in pairs(u) do
    t[value] = key
  end
  return t
end

local function get_layer(layer)
  if layer.name ~= nil then
    if layer.name == 'decoder_attn' then
      decoder_attn = layer
    elseif layer.name:sub(1,3) == 'hop' then
      hop_attn = layer
    elseif layer.name:sub(1,7) == 'softmax' then
      table.insert(softmax_layers, layer)
    elseif layer.name == 'word_vecs_enc' then
      word_vecs_enc = layer
    elseif layer.name == 'word_vecs_dec' then
      word_vecs_dec = layer
    end
  end
end

local function tokens2wordidx(tokens, word2idx, start_symbol)
  local t = {}
  local u = {}
  if start_symbol == 1 then
    table.insert(t, START)
    table.insert(u, START_WORD)
  end

  for _, token in pairs(tokens) do
    local idx = word2idx[token.value] or UNK
    table.insert(t, idx)
    table.insert(u, token.value)
  end
  if start_symbol == 1 then
    table.insert(t, END)
    table.insert(u, END_WORD)
  end
  return torch.LongTensor(t), u
end

local function get_feature_embedding(values, feature2idx, vocab_size, use_lookup)
  if use_lookup == true then
    local t = torch.Tensor(1)
    local idx = feature2idx[values[1]]
    if idx == nil then
      idx = UNK
    end
    t[1] = idx
    return t
  else
    local emb = {}
    for _ = 1, vocab_size do
      table.insert(emb, 0)
    end
    for i = 1, #values do
      local idx = feature2idx[values[i]]
      if idx == nil then
        idx = UNK
      end
      emb[idx] = 1
    end
    return torch.DoubleTensor(emb):view(1,#emb)
  end
end

local function features2featureidx(features, feature2idx, idx2feature,
                                   use_lookup, start_symbol, decoder)
  local out = {}

  if decoder == 1 then
    table.insert(out, {})
    for j = 1, #feature2idx do
      local emb = get_feature_embedding({UNK_WORD}, feature2idx[j], #idx2feature[j], use_lookup[j])
      table.insert(out[#out], emb)
    end
  end

  if start_symbol == 1 then
    table.insert(out, {})
    for j = 1, #feature2idx do
      local emb = get_feature_embedding({START_WORD}, feature2idx[j], #idx2feature[j], use_lookup[j])
      table.insert(out[#out], emb)
    end
  end

  for i = 1, #features do
    table.insert(out, {})
    for j = 1, #feature2idx do
      local emb = get_feature_embedding(features[i][j], feature2idx[j], #idx2feature[j], use_lookup[j])
      table.insert(out[#out], emb)
    end
  end

  if start_symbol == 1 and decoder == 0 then
    table.insert(out, {})
    for j = 1, #feature2idx do
      local emb = get_feature_embedding({END_WORD}, feature2idx[j], #idx2feature[j], use_lookup[j])
      table.insert(out[#out], emb)
    end
  end

  return out
end

local function word2charidx(word, chars_idx, max_word_l, t)
  t[1] = START
  local i = 2
  for _, char in utf8.next, word do
    char = utf8.char(char)
    local char_idx = chars_idx[char] or UNK
    t[i] = char_idx
    i = i+1
    if i >= max_word_l then
      t[i] = END
      break
    end
  end
  if i < max_word_l then
    t[i] = END
  end
  return t
end

local function tokens2charidx(tokens, chars_idx, max_word_l, start_symbol)
  local words = {}
  if start_symbol == 1 then
    table.insert(words, START_WORD)
  end

  for _, token in pairs(tokens) do
    table.insert(words, token.value)
  end
  if start_symbol == 1 then
    table.insert(words, END_WORD)
  end
  local chars = torch.ones(#words, max_word_l)
  for i = 1, #words do
    chars[i] = word2charidx(words[i], chars_idx, max_word_l, chars[i])
  end
  return chars, words
end

local function wordidx2tokens(sent, features, idx2word, idx2feature, source_words, attn)
  local tokens = {}
  local pos = 0

  for i = 2, #sent-1 do -- skip START and END
    local fields = {}
    if sent[i] == UNK and opt.replace_unk == 1 then
      -- retrieve source word with max attention
      local _, max_index = attn[i]:max(1)
      local s = source_words[max_index[1]]

      if phrase_table[s] ~= nil then
        print('Unknown token "' .. s .. '" replaced by source token "' ..phrase_table[s] .. '"')
      end
      local r = phrase_table[s] or s
      table.insert(fields, r)
    else
      table.insert(fields, idx2word[sent[i]])
    end
    for j = 1, model_opt.num_target_features do
      local values = {}
      for k = 1, #features[i][j] do
        table.insert(values, idx2feature[j][features[i][j][k]])
      end
      local values_str = table.concat(values, ',')
      table.insert(fields, values_str)
    end

    local token_value = table.concat(fields, '-|-')
    table.insert(tokens, {
      value = token_value,
      range = {
        begin = pos,
        ['end'] = pos + string.len(token_value)
      },
      attention = attn[i]
    })
    pos = pos + string.len(token_value) + 1
  end

  return tokens
end

local function clean_sent(sent)
  local s = stringx.replace(sent, UNK_WORD, '')
  s = stringx.replace(s, START_WORD, '')
  s = stringx.replace(s, END_WORD, '')
  s = stringx.replace(s, START_CHAR, '')
  s = stringx.replace(s, END_CHAR, '')
  return s
end

local function strip(s)
  return s:gsub("^%s+",""):gsub("%s+$","")
end

local function extract_features(tokens)
  local cleaned_tokens = {}
  local features = {}

  for _, entry in pairs(tokens) do
    local field = entry.value:split('%-|%-')
    local word = clean_sent(field[1])
    if string.len(word) > 0 then
      local cleaned_token = copy(entry)
      cleaned_token.value = word
      table.insert(cleaned_tokens, cleaned_token)

      if #field > 1 then
        table.insert(features, {})
      end

      for i= 2, #field do
        local values = field[i]:split(',')
        table.insert(features[#features], values)
      end
    end
  end

  return cleaned_tokens, features
end

local function build_absolute_paths(resourcesDir)
  local function isempty(s)
    return s == nil or s == ''
  end

  if(not isempty(resourcesDir)) then
    if not isempty(opt.model) then
      opt.model = path.join(resourcesDir, opt.model)
    end
    if not isempty(opt.src_file) then
      opt.src_file = path.join(resourcesDir, opt.src_file)
    end
    if not isempty(opt.targ_file) then
      opt.targ_file = path.join(resourcesDir, opt.targ_file)
    end
    if not isempty(opt.output_file) then
      opt.output_file = path.join(resourcesDir, opt.output_file)
    end
    if not isempty(opt.src_dict) then
      opt.src_dict = path.join(resourcesDir, opt.src_dict)
    end
    if not isempty(opt.targ_dict) then
      opt.targ_dict = path.join(resourcesDir, opt.targ_dict)
    end
    if not isempty(opt.char_dict) then
      opt.char_dict = path.join(resourcesDir, opt.char_dict)
    end
  end
end

local function init(arg, resourcesDir)
  -- parse input params
  opt = cmd:parse(arg)

  build_absolute_paths(resourcesDir)
  assert(path.exists(opt.model), 'model does not exist')

  if opt.gpuid >= 0 then
    require 'cutorch'
    require 'cunn'
    if opt.cudnn == 1 then
      require 'cudnn'
    end
  end
  print('loading ' .. opt.model .. '...')
  local checkpoint = torch.load(opt.model)
  print('done!')

  if opt.replace_unk == 1 then
    phrase_table = {}
    if path.exists(opt.srctarg_dict) then
      local f = io.open(opt.srctarg_dict,'r')
      for line in f:lines() do
        local c = line:split("|||")
        phrase_table[strip(c[1])] = c[2]
      end
    end
  end

  -- load model and word2idx/idx2word dictionaries
  model, model_opt = checkpoint[1], checkpoint[2]
  for i = 1, #model do
    model[i]:evaluate()
  end
  -- for backward compatibility
  model_opt.brnn = model_opt.brnn or 0
  model_opt.input_feed = model_opt.input_feed or 1
  model_opt.attn = model_opt.attn or 1
  model_opt.num_source_features = model_opt.num_source_features or 0
  model_opt.num_target_features = model_opt.num_target_features or 0
  info = checkpoint[3]    
  if opt.src_dict == "" then
    idx2word_src = checkpoint[4]
  else
    idx2word_src = idx2key(opt.src_dict)
  end
  word2idx_src = flip_table(idx2word_src)  

  if opt.targ_dict == "" then
    idx2word_targ = checkpoint[5]
  else
    idx2word_targ = idx2key(opt.targ_dict)
  end
  word2idx_targ = flip_table(idx2word_targ)  

  if opt.feature_dict_prefix == "" then
    idx2feature_src = checkpoint[6]
    idx2feature_targ = checkpoint[7]
  else
    idx2feature_src = {}
    idx2feature_targ = {}
    for i = 1, model_opt.num_source_features do
      table.insert(idx2feature_src, idx2key(opt.feature_dict_prefix .. '.source_feature_' .. i .. '.dict'))
    end
    for i = 1, model_opt.num_target_features do
      table.insert(idx2feature_targ, idx2key(opt.feature_dict_prefix .. '.target_feature_' .. i .. '.dict'))
    end
  end
  for i = 1, model_opt.num_source_features do
    table.insert(feature2idx_src, flip_table(idx2feature_src[i]))
  end
  for i = 1, model_opt.num_target_features do
    table.insert(feature2idx_targ, flip_table(idx2feature_targ[i]))
  end

  if opt.char_dict == "" then
    idx2char = checkpoint[8]
  else
    idx2char = idx2key(opt.char_dict)
  end
  char2idx = flip_table(idx2char)    

  if model_opt.source_features_lookup == nil then
    model_opt.source_features_lookup = {}
    for _ = 1, model_opt.num_source_features do
      table.insert(model_opt.source_features_lookup, false)
    end
    model_opt.target_features_lookup = {}
    for _ = 1, model_opt.num_target_features do
      table.insert(model_opt.target_features_lookup, false)
    end
  end

  -- load character dictionaries if needed
  if model_opt.use_chars_enc == 1 or model_opt.use_chars_dec == 1 then
    utf8 = require 'lua-utf8'
    char2idx = flip_table(idx2key(opt.char_dict))
    model[1]:apply(get_layer)

    if model_opt.use_chars_dec == 1 then
      word2charidx_targ = torch.LongTensor(#idx2word_targ, model_opt.max_word_l):fill(PAD)
      for i = 1, #idx2word_targ do
        word2charidx_targ[i] = word2charidx(idx2word_targ[i], char2idx,
          model_opt.max_word_l, word2charidx_targ[i])
      end
    end
  end

  if opt.gpuid >= 0 then
    cutorch.setDevice(opt.gpuid)
    for i = 1, #model do
      if opt.gpuid2 >= 0 then
        if i == 1 or i == 4 then
          cutorch.setDevice(opt.gpuid)
        else
          cutorch.setDevice(opt.gpuid2)
        end
      end
      model[i]:double():cuda()
      model[i]:evaluate()
    end
  end

  softmax_layers = {}
  model[2]:apply(get_layer)
  local attn_layer
  if model_opt.attn == 1 then
    decoder_attn:apply(get_layer)
    decoder_softmax = softmax_layers[1]
    attn_layer = torch.zeros(opt.beam, opt.max_sent_l)
  end

  context_proto = torch.zeros(1, opt.max_sent_l, model_opt.rnn_size)
  local h_init_dec = torch.zeros(opt.beam, model_opt.rnn_size)
  local h_init_enc = torch.zeros(1, model_opt.rnn_size)
  if opt.gpuid >= 0 then
    h_init_enc = h_init_enc:cuda()
    h_init_dec = h_init_dec:cuda()
    cutorch.setDevice(opt.gpuid)
    if opt.gpuid2 >= 0 then
      cutorch.setDevice(opt.gpuid)
      context_proto = context_proto:cuda()
      cutorch.setDevice(opt.gpuid2)
      context_proto2 = torch.zeros(opt.beam, opt.max_sent_l, model_opt.rnn_size):cuda()
    else
      context_proto = context_proto:cuda()
    end
    if model_opt.attn == 1 then
      attn_layer:cuda()
    end
  end
  init_fwd_enc = {}
  init_fwd_dec = {}
  if model_opt.input_feed == 1 then
    table.insert(init_fwd_dec, h_init_dec:clone())
  end

  for _ = 1, model_opt.num_layers do
    table.insert(init_fwd_enc, h_init_enc:clone())
    table.insert(init_fwd_enc, h_init_enc:clone())
    table.insert(init_fwd_dec, h_init_dec:clone()) -- memory cell
    table.insert(init_fwd_dec, h_init_dec:clone()) -- hidden state
  end

  State = StateAll
end


--[[
  @table token

  @field value - token value (word)
  @field [attention] - attention tensor used for predicting the token
  @field [range] - token range:
    {
      begin = <begin char index>,
      end = <end char index (not including)>
    }
]]

--[[
  @function search
  @brief Performs a beam search.

  @param tokens - array of input tokens
  @param gold - array of reference tokens for calculating gold scores
  @return pred_tokens - array of predicted tokens
  @return info - table containing various info:
    {
      pred_score = <prediction score>,
      pred_words = <prediction words count>,
      gold_score = <gold score>,
      gold_words = <gold words count>,
      nbests = [
        {
          tokens = <array of tokens>,
          score = <prediction score>
        }
      ]
    }
]]
local function search(tokens, gold)
  local cleaned_tokens, source_features_str = extract_features(tokens)
  local source, source_str
  local source_features = features2featureidx(source_features_str, feature2idx_src, idx2feature_src, model_opt.source_features_lookup, model_opt.start_symbol, 0)
  if model_opt.use_chars_enc == 0 then
    source, source_str = tokens2wordidx(cleaned_tokens, word2idx_src, model_opt.start_symbol)
  else
    source, source_str = tokens2charidx(cleaned_tokens, char2idx, model_opt.max_word_l, model_opt.start_symbol)
  end

  local target
  local target_features
  if gold ~= nil then
    local gold_tokens, target_features_str = extract_features(gold)
    target = tokens2wordidx(gold_tokens, word2idx_targ, 1)
    target_features = features2featureidx(target_features_str, feature2idx_targ, idx2feature_targ, model_opt.target_features_lookup, 1, 1)
  end

  local pred, pred_features, pred_score, attn, gold_score, all_sents, all_scores, all_attn = generate_beam(
    opt.beam, opt.max_sent_l, source, source_features, target, target_features)

  local pred_tokens = wordidx2tokens(pred, pred_features, idx2word_targ, idx2feature_targ, source_str, attn)

  local info = {
    nbests = {},
    pred_score = pred_score,
    pred_words = #pred - 1
  }

  if gold ~= nil then
    info.gold_score = gold_score
    info.gold_words = target:size(1) - 1
  end

  if opt.n_best > 1 and model_opt.num_target_features == 0 then
    for n = 1, opt.n_best do
      local pred_tokens_n = wordidx2tokens(all_sents[n], pred_features, idx2word_targ, idx2feature_targ, source_str, all_attn[n])
      table.insert(info.nbests, {
        tokens = pred_tokens_n,
        score = all_scores[n]
      })
    end
  end

  return pred_tokens, info
end

return {
  init = init,
  search = search,
  getOptions = function() return opt end
}
