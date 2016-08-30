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
