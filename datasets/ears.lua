--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--

local M = {}
local EarDataset = torch.class('resnet.EarDataset', M)

require 'image'
require 'paths'
require 'math'

function EarDataset:__init(imageInfo, opt, split)
   assert(imageInfo[split], split)
   self.imageInfo = imageInfo[split]
   self.split = split
   self.sourcedir = 'datasets/wheat'
end

function EarDataset:get(i)
   local image = image.load(paths.concat(self.sourcedir, self.imageInfo[i].filename))
   local label = self.imageInfo[i].annotation

   assert(#label.ears > 0, "Error, no ears found in " .. self.imageInfo[i].filename)
   -- Chose a single ear for this request of the image
   local idx = math.random(#label.ears)
   local ear = label.ears[idx]

   return {
      input = image,
      target = { ear = ear:clone(), tips = label.tips:clone(), spikelets = label.spikelets:clone(), awns = label.awns }
   }
end

function EarDataset:size()
   return #self.imageInfo
end

return M.EarDataset
