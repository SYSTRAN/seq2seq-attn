local beam = require 's2sa.beam'

function init(arg, resourcesDir)
  beam.init(arg, resourcesDir)
end

function translate(input)
  return beam.search(input)
end
