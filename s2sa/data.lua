--
-- Manages encoder/decoder data matrices.
--

function features_per_timestep(features)
  local data = {}

  if #features ~= 0 then
    local sent_len = features[1]:size(1)
    for i = 1,sent_len do
      table.insert(data, {})
      for j = 1,#features do
        table.insert(data[i], features[j][i])
      end
    end
  end

  return data
end

function features_on_gpu(features)
  local clone = {}
  for i = 1,#features do
    for j = 1,#features[i] do
      table.insert(clone[i], features[i][j]:cuda())
    end
  end
  return clone
end

function find_powers_of_2(value)
  local powers = {}
  local i = 1
  while i <= value do
    if bit.band(i, value) ~= 0 then
      table.insert(powers, math.log(i)/math.log(2))
    end
    i = bit.lshift(i, 1)
  end
  return powers
end

function bitfield_to_ids(values, id_max_size)
  local powers = {}
  for k = 1, values:size(1) do
    local pows = find_powers_of_2(values[k])
    table.insert(powers, pows)
  end

  local ids = {}
  local offset = 0
  for k = 1, #powers do
    for i = 1, #powers[k] do
      local id = powers[k][i]+1+offset
      table.insert(ids, id)
    end
    offset = offset + id_max_size
  end

  return ids
end

function bitfield_to_vec(values, size, id_max_size)
  local ids = bitfield_to_ids(values, id_max_size)
  local vec = torch.Tensor(size):zero()
  for i = 1, #ids do
    vec[ids[i]] = 1.0 / #ids
  end
  return vec
end

function generate_vecs(features, sizes, use_lookup, id_max_size)
  local data = {}
  for i = 1,#features do
    table.insert(data, {})
    for j = 1,#features[i] do
      local batch_size = features[i][j]:size(1)
      local t
      if use_lookup[j] == false then
        t = torch.Tensor(batch_size, sizes[j])
        for k = 1, batch_size do
          t[k]:copy(bitfield_to_vec(features[i][j][k], sizes[j], id_max_size))
        end
      else
        t = torch.Tensor(batch_size)
        for k = 1, batch_size do
          t[k] = bitfield_to_ids(features[i][j][k], id_max_size)[1]
        end
      end
      table.insert(data[i], t)
    end
  end
  return data
end

local data = torch.class("data")

function data:__init(opt, data_file)
  local f = hdf5.open(data_file, 'r')

  self.source = f:read('source'):all()
  self.target = f:read('target'):all()
  self.target_output = f:read('target_output'):all()
  self.target_l = f:read('target_l'):all() --max target length each batch
  self.target_l_all = f:read('target_l_all'):all()
  self.target_l_all:add(-1)
  self.batch_l = f:read('batch_l'):all()
  self.source_l = f:read('batch_w'):all() --max source length each batch

  self.num_source_features = f:read('num_source_features'):all()[1]
  self.num_target_features = f:read('num_target_features'):all()[1]
  self.identifier_max_size = f:read('identifier_max_size'):all()[1]
  self.source_features = {}
  self.target_features = {}
  self.target_features_output = {}
  self.source_features_size = {}
  self.target_features_size = {}
  self.source_features_vec_size = {}
  self.target_features_vec_size = {}
  self.source_features_use_lookup = {}
  self.target_features_use_lookup = {}
  self.total_source_features_size = 0
  self.total_target_features_size = 0

  for i = 1,self.num_source_features do
    table.insert(self.source_features, f:read('source_feature_' .. i):all())
    local max_values = f:read('source_feature_' .. i .. '_max_values'):all()[1]
    local feature_size = f:read('source_feature_' .. i .. '_size'):all()[1]
    local use_lookup = (max_values == 1 and opt.feature_embeddings == 1)
    table.insert(self.source_features_use_lookup, use_lookup)
    table.insert(self.source_features_size, feature_size)
    if use_lookup == true then
      feature_size = math.floor(feature_size * opt.feature_embeddings_dim_factor)
    end
    table.insert(self.source_features_vec_size, feature_size)
    self.total_source_features_size = self.total_source_features_size + feature_size
  end
  for i = 1,self.num_target_features do
    table.insert(self.target_features, f:read('target_feature_' .. i):all())
    table.insert(self.target_features_output,
      f:read('target_feature_output_' .. i):all())
    local max_values = f:read('target_feature_' .. i .. '_max_values'):all()[1]
    local feature_size = f:read('target_feature_' .. i .. '_size'):all()[1]
    local use_lookup = (max_values == 1 and opt.feature_embeddings == 1)
    table.insert(self.target_features_use_lookup, use_lookup)
    table.insert(self.target_features_size, feature_size)
    if use_lookup == true then
      feature_size = math.floor(feature_size * opt.feature_embeddings_dim_factor)
    end
    table.insert(self.target_features_vec_size, feature_size)
    self.total_target_features_size = self.total_target_features_size + feature_size
  end

  if opt.start_symbol == 0 then
    self.source_l:add(-2)
    self.source = self.source[{{},{2, self.source:size(2)-1}}]
    for i = 1,self.num_source_features do
      self.source_features[i] = self.source_features[i][{{},{2, self.source_features[i]:size(2)-1}}]
    end
  end
  self.batch_idx = f:read('batch_idx'):all()

  self.target_size = f:read('target_size'):all()[1]
  self.source_size = f:read('source_size'):all()[1]
  self.target_nonzeros = f:read('target_nonzeros'):all()

  if opt.use_chars_enc == 1 then
    self.source_char = f:read('source_char'):all()
    self.char_size = f:read('char_size'):all()[1]
    self.char_length = self.source_char:size(3)
    if opt.start_symbol == 0 then
      self.source_char = self.source_char[{{}, {2, self.source_char:size(2)-1}}]
    end
  end

  if opt.use_chars_dec == 1 then
    self.target_char = f:read('target_char'):all()
    self.char_size = f:read('char_size'):all()[1]
    self.char_length = self.target_char:size(3)
  end

  self.length = self.batch_l:size(1)
  self.seq_length = self.target:size(2)
  self.batches = {}
  local max_source_l = self.source_l:max()
  local source_l_rev = torch.ones(max_source_l):long()
  for i = 1, max_source_l do
    source_l_rev[i] = max_source_l - i + 1
  end
  for i = 1, self.length do
    local source_i, target_i
    local source_features_i, target_features_i, target_features_output_i

    local target_output_i = self.target_output:sub(self.batch_idx[i],self.batch_idx[i]
      +self.batch_l[i]-1, 1, self.target_l[i]):transpose(1,2)
    local target_l_i = self.target_l_all:sub(self.batch_idx[i],
      self.batch_idx[i]+self.batch_l[i]-1)
    if opt.use_chars_enc == 1 then
      source_i = self.source_char:sub(self.batch_idx[i],
        self.batch_idx[i] + self.batch_l[i]-1, 1,
        self.source_l[i]):transpose(1,2):contiguous()
    else
      source_i = self.source:sub(self.batch_idx[i], self.batch_idx[i]+self.batch_l[i]-1,
        1, self.source_l[i]):transpose(1,2)
    end
    if opt.reverse_src == 1 then
      source_i = source_i:index(1, source_l_rev[{{max_source_l-self.source_l[i]+1,
            max_source_l}}])
    end

    if opt.use_chars_dec == 1 then
      target_i = self.target_char:sub(self.batch_idx[i],
        self.batch_idx[i] + self.batch_l[i]-1, 1,
        self.target_l[i]):transpose(1,2):contiguous()
    else
      target_i = self.target:sub(self.batch_idx[i], self.batch_idx[i]+self.batch_l[i]-1,
        1, self.target_l[i]):transpose(1,2)
    end

    local source_feats = {}
    local target_feats = {}
    local target_feats_output = {}

    for j = 1,self.num_source_features do
      table.insert(source_feats,
        self.source_features[j]:sub(self.batch_idx[i],
          self.batch_idx[i]+self.batch_l[i]-1,
          1, self.source_l[i]):transpose(1,2):double())
      if opt.reverse_src == 1 then
        source_feats[j] = source_feats[j]:index(1, source_l_rev[{{max_source_l-self.source_l[i]+1,
              max_source_l}}])
      end
    end

    for j = 1,self.num_target_features do
      table.insert(target_feats_output,
        self.target_features_output[j]:sub(self.batch_idx[i],
          self.batch_idx[i]+self.batch_l[i]-1,
          1, self.target_l[i]):transpose(1,2):double())
      table.insert(target_feats,
        self.target_features[j]:sub(self.batch_idx[i],
          self.batch_idx[i]+self.batch_l[i]-1,
          1, self.target_l[i]):transpose(1,2):double())
    end

    -- convert table of timesteps per feature to a table of features per timestep
    source_features_i = features_per_timestep(source_feats)
    target_features_i = features_per_timestep(target_feats)
    target_features_output_i = features_per_timestep(target_feats_output)

    table.insert(self.batches, {target_i,
        target_output_i,
        target_output_i:transpose(1,2),
        self.target_nonzeros[i],
        source_i,
        self.batch_l[i],
        self.target_l[i],
        self.source_l[i],
        target_l_i,
        source_features_i,
        target_features_i,
        target_features_output_i})
  end
end

function data:size()
  return self.length
end

function data.__index(self, idx)
  if type(idx) == "string" then
    return data[idx]
  else
    local target_input = self.batches[idx][1]
    local target_output = self.batches[idx][2]
    local nonzeros = self.batches[idx][3]
    local source_input = self.batches[idx][4]
    local batch_l = self.batches[idx][5]
    local target_l = self.batches[idx][6]
    local source_l = self.batches[idx][7]
    local target_l_all = self.batches[idx][8]
    local source_features = generate_vecs(self.batches[idx][9],
                                          self.source_features_size,
                                          self.source_features_use_lookup,
                                          self.identifier_max_size)
    local target_features = generate_vecs(self.batches[idx][10],
                                          self.target_features_size,
                                          self.target_features_use_lookup,
                                          self.identifier_max_size)
    local target_features_output = generate_vecs(self.batches[idx][11],
                                                 self.target_features_size,
                                                 self.target_features_use_lookup,
                                                 self.identifier_max_size)
    if opt.gpuid >= 0 then --if multi-gpu, source lives in gpuid1, rest on gpuid2
      cutorch.setDevice(opt.gpuid)
      source_input = source_input:cuda()
      source_features = features_on_gpu(source_features)
      if opt.gpuid2 >= 0 then
        cutorch.setDevice(opt.gpuid2)
      end
      target_input = target_input:cuda()
      target_output = target_output:cuda()
      target_l_all = target_l_all:cuda()
      target_features = features_on_gpu(target_features)
      target_features_output = features_on_gpu(target_features_output)
    end
    return {target_input, target_output, nonzeros, source_input,
      batch_l, target_l, source_l, target_l_all,
      source_features, target_features, target_features_output}
  end
end

return data
