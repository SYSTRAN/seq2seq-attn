require 'nn'
require 'string'
require 'hdf5'
require 'nngraph'
require 'cunn'
require 'cutorch'

require 's2sa.models'
require 's2sa.data'

cmd = torch.CmdLine()
-- file location
cmd:option('-model', 'model.t7','model file')
opt = cmd:parse(arg)

function main()
   print('loading model ' .. opt.model)
   checkpoint = torch.load(opt.model)
   model, model_opt, info = checkpoint[1], checkpoint[2], checkpoint[3]
   print(model_opt)
   print(info)   
end

main()

