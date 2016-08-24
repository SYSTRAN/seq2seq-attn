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
end

-- usePrealloc is based on the same principle but use pre-allocated memory at the beginning of the process
preallocWarning={}
preallocTable={}

-- applyFor specify if applies for output (O) or gradientInput (G)
function nn.Module:usePrealloc(preallocName, applyFor)
   applyFor = applyFor or "GO"
   if string.find(applyFor,"G") then
    self.preallocGradient=preallocName
   end
   if string.find(applyFor,"O") then
    self.preallocOutput=preallocName
   end
   return self
end

function nn.Module:setPrealloc()
   if self.preallocOutput then
     if preallocTable[self.preallocOutput]==nil then
       if not(preallocWarning[self.prealloc]) then
         print('WARNING: no prealloc memory defined for \'' .. self.prealloc .. '\'')
         preallocWarning[self.preallocOutput]=1
       end
       return
     end
    self.output = preallocTable[self.preallocOutput]
   end
   if self.preallocGradient then
     if preallocTable[self.preallocGradient]==nil then
       if not(preallocWarning[self.prealloc]) then
         print('WARNING: no prealloc memory defined for \'' .. self.prealloc .. '\'')
         preallocWarning[self.preallocGradient]=1
       end
       return
     end
      self.gradInput = preallocTable[self.preallocGradient]
   end
end
