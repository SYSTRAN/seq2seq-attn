-- module for memory management

-- reuseMem is used for reusing output tensor for storing gradInput and optimizing memory allocation
-- use :reuseMem() on the module to allow the feature
-- then apply setReuse after initialization
function nn.Module:reuseMem()
  self.reuse = true
  return self
end

function nn.Module:setReuse()
  if self.reuse then
    self.gradInput = self.output
  end
  return self
end

-- usePrealloc is based on the same principle but use pre-allocated memory at the beginning of the process
preallocWarning = {}
preallocTable = {}

function nn.Module:usePrealloc(preallocName)
  self.prealloc = preallocName
  return self
end

function nn.Module:setPrealloc()
  if self.prealloc then
    if preallocTable[self.prealloc] == nil then
      if not(preallocWarning[self.prealloc]) then
        print('WARNING: no prealloc memory defined for \'' .. self.prealloc .. '\'')
        preallocWarning[self.prealloc] = 1
      end
      return
    end
    self.gradInput = preallocTable[self.prealloc]
  end
  return self
end

-- temp is even more powerful, it provides a method to call as soon as the input/output of the object
-- can be freed

function nn.Module:temp()
  self.isTemp = true
  return self
end

function clearTensors(o)
  if o == nil then
    return nil
  end
  if type(o) == "table" then
    for i = 1,#o do
      o[i] = clearTensors(o[i])
    end
    return o
  end
  local ttype=o:type()
  if string.find(ttype,"Tensor") then
    if string.find(ttype,"Cuda") then
      if o:getDevice() ~= 0 then
        -- we need to allocate at least one byte, otherwise we forget on which device it is
        -- since unallocated tensors are not attached to devices
        cutorch.setDevice(o:getDevice())
        return torch.CudaTensor(1)
      else
        -- otherwise, it was a CudaTensor without size, we don't change anything
        return o
      end
    else
      return torch.Tensor()
    end
  end
  return o
end

function nn.Module:cleanTemp()
  if self.isTemp then
    self.output = clearTensors(self.output)
    self.gradInput = clearTensors(self.gradInput)
  end
  return self
end
