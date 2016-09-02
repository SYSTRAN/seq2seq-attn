local beam = require 's2sa.beam'
local path = require 'pl.path'

local function sent2tokens(line)
  local tokens = {}
  for tok in line:gmatch'([^%s]+)' do
    table.insert(tokens, tok)
  end
  return tokens
end

local function annotations2sent(annotations)
  local tokens = {}
  for _, annotation in pairs(annotations) do
    table.insert(tokens, annotation.value)
  end

  return table.concat(tokens, ' ')
end

local function main()
  beam.init(arg)
  local opt = beam.getOptions()

  assert(path.exists(opt.src_file), 'src_file does not exist')

  local file = io.open(opt.src_file, "r")
  local out_file = io.open(opt.output_file,'w')

  local sent_id = 0
  local gold = {}
  local pred_score_total = 0
  local gold_score_total = 0
  local pred_words_total = 0
  local gold_words_total = 0

  -- load gold labels if it exists
  if path.exists(opt.targ_file) then
    print('loading GOLD labels at ' .. opt.targ_file)

    local targ_file = io.open(opt.targ_file, 'r')
    for line in targ_file:lines() do
      table.insert(gold, sent2tokens(line))
    end
  else
    opt.score_gold = 0
  end

  for line in file:lines() do
    sent_id = sent_id + 1
    print('SENT ' .. sent_id .. ': ' .. line)

    if opt.score_gold == 1 then
      print('GOLD ' .. sent_id .. ': ' .. table.concat(gold[sent_id], ' '))
    end

    local annotations, info = beam.search(sent2tokens(line), gold[sent_id])
    local sent = annotations2sent(annotations)
    print('PRED ' .. sent_id .. ': '..sent)
    out_file:write(sent .. '\n')

    pred_score_total = pred_score_total + info.pred_score
    pred_words_total = pred_words_total + info.pred_words

    if opt.score_gold == 1 then
      print(string.format("PRED SCORE: %.4f, GOLD SCORE: %.4f", info.pred_score, info.gold_score))
      gold_score_total = gold_score_total + info.gold_score
      gold_words_total = gold_words_total + info.gold_words
    end

    for n = 1, #info.nbests do
      local nbest = annotations2sent(info.nbests[n].annotations)
      local out_n = string.format("%d ||| %s ||| %.4f", n, nbest, info.nbests[n].score)
      print(out_n)
      out_file:write(nbest .. '\n')
    end

    print('')
  end

  print(string.format("PRED AVG SCORE: %.4f, PRED PPL: %.4f", pred_score_total / pred_words_total,
    math.exp(-pred_score_total/pred_words_total)))
  if opt.score_gold == 1 then
    print(string.format("GOLD AVG SCORE: %.4f, GOLD PPL: %.4f",
      gold_score_total / gold_words_total,
      math.exp(-gold_score_total/gold_words_total)))
  end
  out_file:close()
end

main()
