local PSNRCriterion, parent = torch.class('nn.PSNRCriterion', 'nn.Criterion')

function PSNRCriterion:__init(sizeAverage)
   parent.__init(self)
   if sizeAverage ~= nil then
     self.sizeAverage = sizeAverage
   else
     self.sizeAverage = false
   end
    self.effectLossLowerBound = 0.5

end

function PSNRCriterion:updateOutput(input, target)
   self.output_tensor = self.output_tensor or input.new(1)
   input.THNN.MSECriterion_updateOutput(
      input:cdata(),
      target:cdata(),
      self.output_tensor:cdata(),
      self.sizeAverage
   )
   self.output = self.output_tensor[1]
   self.output=-10* math.log(255.0*255.0/self.output)/math.log(10)
   
   
   return self.output
end

function PSNRCriterion:updateGradInput(input, target)
   input.THNN.MSECriterion_updateGradInput(
      input:cdata(),
      target:cdata(),
      self.gradInput:cdata(),
      self.sizeAverage
   )
 
   self.gradInput = 10*self.gradInput/( math.log(10)*self.output_tensor[1])

   return self.gradInput
end
