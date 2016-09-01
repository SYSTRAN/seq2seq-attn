local beam = require 's2sa.beam'

function init(arg, resourcesDir)
  beam.init(arg, resourcesDir)
end

function translate(tokens)
  return beam.search(tokens)
end
