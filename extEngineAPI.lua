local beam = require 's2sa.beam'


local function extract_tokens(tokens)
  local source_tokens = {}
  for _, tok in pairs(tokens) do
    table.insert(source_tokens, tok.value)
  end

  return source_tokens
end

--[[
Build alignment between source and target tokens

Each target token is aligned with the source token that had the highest attention weight
when this target token was selected by the beam search.
]]
local function build_alignment(source_annotations, target_annotations)
  local alignment = {}

  for _, target_annotation in pairs(target_annotations) do
    local _, max_index = target_annotation.attention:max(1)
    local source_index = max_index[1]
    if source_index ~= nil and source_index <= #source_annotations then
      local source_annotation = source_annotations[source_index]
      table.insert(alignment, {
        source = source_annotation.range,
        target = target_annotation.range
      })
    end
  end

  return alignment
end


--[[
API exposed to the Lua ExtEngine library:

  init(arg, resourcesDir)
  translatesource_annotations)
]]
function init(arg, resources_dir)
  beam.init(arg, resources_dir)
end

function translate(source_annotations)
  local source_tokens = extract_tokens(source_annotations)
  local target_annotations = beam.search(source_tokens)

  local alignment = build_alignment(source_annotations, target_annotations)

  return target_annotations, alignment
end
