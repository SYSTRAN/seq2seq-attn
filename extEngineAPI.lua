local beam = require 's2sa.beam'


--[[
Build alignment between source and target tokens

Each target token is aligned with the source token that had the highest attention weight
when this target token was selected by the beam search.
]]
local function build_alignment(source_tokens, target_tokens)
  local alignment = {}

  for _, target_token in pairs(target_tokens) do
    local _, max_index = target_token.attention:max(1)
    local source_index = max_index[1]
    if source_index ~= nil and source_index <= #source_tokens then
      local source_token = source_tokens[source_index]
      table.insert(alignment, {
        source = source_token.range,
        target = target_token.range
      })
    end
  end

  return alignment
end


--[[
API exposed to the Lua ExtEngine library:

  init(arg, resourcesDir)
  translate(source_tokens)
]]
function init(arg, resources_dir)
  beam.init(arg, resources_dir)
end

function translate(source_tokens)
  local target_tokens = beam.search(source_tokens)

  local alignment = build_alignment(source_tokens, target_tokens)

  return target_tokens, alignment
end
