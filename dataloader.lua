--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Multi-threaded data loader
--

local datasets = require 'datasets/init'
local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')

local M = {}
local DataLoader = torch.class('resnet.DataLoader', M)

function DataLoader.create(opt)
    -- The train and val loader
    local loaders = {}

    for i, split in ipairs{'train', 'valid'} do
        local dataset = datasets.create(opt, split)
        loaders[i] = M.DataLoader(dataset, opt, split)
    end
    return table.unpack(loaders)
end

function DataLoader:__init(dataset, opt, split)
    local manualSeed = opt.manualSeed
    local function init()
        torch.setdefaulttensortype('torch.FloatTensor')
        require('datasets/' .. opt.dataset)
    end
    local function main(idx)
        local utils = require('dataloaderutil')
        
        if manualSeed ~= 0 then
         torch.manualSeed(manualSeed + idx)
        end
        torch.setnumthreads(1)
        _G.dataset = dataset
        -- TODO - move this into utils
        --_G.preprocess = dataset:preprocess()
        local function rnd(x) return math.max(-2*x,math.min(2*x,torch.randn(1)[1]*x)) end
        _G.rnd = rnd
        _G.renderheatmap = utils.renderheatmap

        local function getRotationTransform(rotcenter, rot)
            local t = torch.eye(3)
            local cosr = math.cos(rot)
            local sinr = math.sin(rot)
            local tx, ty = rotcenter[1],rotcenter[2]
            t[1][3] = -tx
            t[2][3] = -ty
            local ti = torch.eye(3)
            ti[1][3] = tx
            ti[2][3] = ty
            r = torch.eye(3)
            r[1][1] = cosr
            r[1][2] = -sinr
            r[2][1] = sinr
            r[2][2] = cosr

            return ti * r * t
        end

        local function getTranslationScaleTransform(translation, scale)
            local t = torch.eye(3)
            local tx, ty = translation[1],translation[2]
            t[1][3] = -tx
            t[2][3] = -ty
            
            local s = torch.eye(3)
            s[1][1] = scale
            s[2][2] = scale

            return s * t
        end

        _G.getRotationTransform = getRotationTransform
        _G.getTranslationScaleTransform = getTranslationScaleTransform

        return dataset:size()
    end
    local threads, sizes = Threads(opt.nThreads, init, main)
    self.threads = threads
    self.__size = sizes[1][1]
    self.split = split
    self.batchSize = opt[split .. 'Batch']
end

function DataLoader:size()
    return math.floor(self.__size / self.batchSize)
end

function DataLoader:datasetsize()
    return math.floor(self.__size)
end

function DataLoader:batchsize()
    return math.floor(self.batchSize)
end

function DataLoader:run()
    local threads = self.threads
    local size, batchSize = self.__size, self.batchSize
    local perm = torch.randperm(size)

    local idx, sample = 1, nil
    local function enqueue()
        while idx <= size and threads:acceptsjob() do
         local indices = perm:narrow(1, idx, math.min(batchSize, size - idx + 1))
         threads:addjob(
            function(indices, useAugmentation, opt)
                local sz = indices:size(1)
                local inputbatch, imageSize
                local heatmapbatch, targetSize
                local targettable
                local normlengths
                local batchawns

                -- Fixed capture and augmentation parameters
                local captureDim = opt.captureRes
                local inputDim = opt.inputRes
                local outputDim = opt.outputRes
                local captureScale = inputDim / captureDim
                local outputScale = outputDim / inputDim
                local capturePad = opt.capturePad
                local randomCrop = opt.randomCrop ~= 0
                local randomFlip = opt.randomFlip ~= 0
                local gaussSD = opt.hmGauss

                for i, idx in ipairs(indices:totable()) do
                    
                    -- Get next image from dataset
                    local sample = _G.dataset:get(idx)
                    
                    local input = sample.input
                    local target = sample.target
                    local ear = target.ear
                    local tips = target.tips
                    local spkt = target.spikelets
                    local awns = target.awns

                    -- Convert from 0-based annotations to 1-based LUA
                    ear:add(1)
                    tips:add(1)
                    spkt:add(1)

                    local earlength = torch.sqrt((ear[1] - ear[-1]):pow(2):sum())

                    -- Random augmentation parameters
                    local imgwidth, imgheight = input:size(3), input:size(2)

                    local augScale = math.min(math.max(2 ^ _G.rnd(opt.scale),0.5),2.0) -- Sanity check on size
                    local augRotate =  math.min(math.max(_G.rnd(opt.rotate),-0.52),0.52) -- Clamped to +- 30 Degrees
                    --if torch.uniform() < 0.5 then augRotate = 0 end
                    local flip = false
                    if torch.uniform() < 0.5 and randomFlip then flip = true end

                    if useAugmentation ~= true then
                        augScale = 1;
                        augRotate = 1;
                    end

                    -- Calculate initial bounds
                    local y = ear[{{},2}]:min() - capturePad
                    local y2 = ear[{{},2}]:max() + capturePad
                    local x = ear[{{},1}]:min() - capturePad
                    local x2 = ear[{{},1}]:max() + capturePad
                    local width = x2 - x;
                    local height = y2 - y;

                    -- Initial crop bounds
                    local cb = torch.Tensor({x, y, width, height});
                    if cb[3] < captureDim and cb[4] < captureDim then
                        cb[1] = cb[1] - torch.round((captureDim - cb[3]) / 2)
                        cb[3] = captureDim
                    
                        cb[2] = cb[2] - torch.round((captureDim - cb[4]) / 2)
                        cb[4] = captureDim
                    else
                        -- If height > width, pad width to match
                        if cb[3] < cb[4] then
                            cb[1] = cb[1] - torch.round((cb[4] - cb[3]) / 2)
                            cb[3] = cb[4]
                        -- If width > height, pad height to match
                        else
                            cb[2] = cb[2] - torch.round((cb[3] - cb[4]) / 2)
                            cb[4] = cb[3]
                        end
                    end

                    -- If crop is larger than capture res, then crop to size.
                    if (cb[3] > captureDim) then
                        local diff = cb[3] - captureDim
                        if randomCrop then 
                            cb[1] = cb[1] + torch.random(0, diff)
                        else
                            cb[1] = cb[1] + torch.round(diff / 2)
                        end
                        cb[3] = captureDim
                    end

                    if (cb[4] > captureDim) then
                        local diff = cb[4] - captureDim
                        if randomCrop then
                            cb[2] = cb[2] + torch.random(0, diff)
                        else
                            cb[2] = cb[2] + torch.round(diff / 2)
                        end
                        cb[4] = captureDim
                    end

                    -- If random scaling, scale now
                    if (useAugmentation and augScale ~= 1) then
                        local scaledDim = captureDim * augScale;
                        local diff = scaledDim - captureDim
                        cb[{{1,2}}]:add(-(diff/2))
                        cb[{{3,4}}]:mul(augScale)
                    end

                    -- Round cb into pixel coordinates
                    cb:round()

                    -- Clone to rotated bounds
                    local rcb = cb:clone()

                    -- If rotating, add necessary padding
                    local rotatePad = 0
                    if (useAugmentation and augRotate ~= 0) then
                        rotatePad = math.ceil(rcb[3] * (math.sqrt(2) * math.sin((math.pi) / 4 + math.abs(augRotate)) / 2 - 0.5))
                        rcb[{{1,2}}]:add(-rotatePad)
                        rcb[{{3,4}}]:add(2 * rotatePad)
                    end

                    -- Standard bounds checks
                    if rcb[1] < 1 then rcb[1] = 1 end
                    if rcb[2] < 1 then rcb[2] = 1 end

                    if rcb[1] + rcb[3] > imgwidth then rcb[1] = imgwidth - rcb[3] end
                    if rcb[2] + rcb[4] > imgheight then rcb[2] = imgheight - rcb[4] end

                    -- Reset CB in case of bound check movement
                    cb[{{1,2}}] = torch.add(rcb[{{1,2}}], rotatePad)
                    cb[{{3,4}}] = torch.add(rcb[{{3,4}}], -2 * rotatePad)

                    -- Bound assertions
                    assert(rcb[1] >= 1 and rcb[2] >= 1 and rcb[1] + rcb[3] <= imgwidth and rcb[2] + rcb[4] <= imgheight, "Bounds failure, cannot crop.")

                    -- Square assertion
                    assert(rcb[3] == rcb[4], "Crop region is not square.")

                    -- Calculate rotation centre
                    local rotationCentre = torch.Tensor({ cb[1] + cb[3] / 2, cb[2] + cb[4] / 2})
                    
                    -- RCB crop.
                    input = input[{ {}, {rcb[2], rcb[2] + rcb[4] - 1}, {rcb[1], rcb[1] + rcb[3] - 1} }]

                    -- If rotation is required, perform a rotation and then an additional crop to CB.
                    if (useAugmentation and augRotate ~= 0) then
                        input = image.rotate(input, augRotate, 'bilinear')
                        input = input[{ {}, {1 + rotatePad, rcb[4] - rotatePad }, {1 + rotatePad, rcb[3] - rotatePad} }]
                    end

                    -- Scale to input resolution
                    input = image.scale(input, inputDim, inputDim, 'bilinear')

                    if useAugmentation and flip then
                        input = image.hflip(input)
                    end

                    local worldscale = (1/augScale) * captureScale * outputScale
                    local scalemat = _G.getTranslationScaleTransform(cb[{{1,2}}], worldscale)
                    local mat = scalemat
                    earlength = earlength * worldscale

                    -- Add rotation if being used
                    if useAugmentation and augRotate ~= 0 then
                        local rotationmat = _G.getRotationTransform(rotationCentre, -augRotate)
                        mat = mat * rotationmat
                    end

                    local transformedTips = (mat * tips:cat(torch.ones(tips:size(1))):t()):t():ceil()[{{},{1,2}}]
                    local transformedSpkt = (mat * spkt:cat(torch.ones(spkt:size(1))):t()):t():ceil()[{{},{1,2}}]

                    -- Flip tips if horizontal flipping in use
                    if useAugmentation and flip then
                        -- Flip X co-ord only
                        transformedTips[{{},{1}}]:mul(-1):add(outputDim + 1)
                        transformedSpkt[{{},{1}}]:mul(-1):add(outputDim + 1)
                    end

                    -- Determine in-bound tips
                    local function inbounds(pt, size)
                        return pt[1] >= 1 and pt[1] <= size and pt[2] >= 1 and pt[2] <= size
                    end

                    local dt = {}
                    for t = 1,transformedTips:size(1) do
                        if inbounds(transformedTips[t], outputDim) then
                            table.insert(dt, t)
                        end
                    end


                    local ds = {}
                    for t = 1,transformedSpkt:size(1) do
                        if inbounds(transformedSpkt[t], outputDim) then
                            table.insert(ds, t)
                        end
                    end

                    local spktGTPoints
                    if #ds > 0 then
                        spktGTPoints = transformedSpkt:index(1,torch.LongTensor(ds))
                    else
                        spktGTPoints = torch.Tensor(0)
                    end

                    local tipsGTPoints
                    if #dt > 0 then
                        tipsGTPoints = transformedTips:index(1,torch.LongTensor(dt))
                    else
                        tipsGTPoints = torch.Tensor(0)
                    end
                    
                    local hm = torch.zeros(opt.nOutChannels, outputDim, outputDim);
                    hm[1] = _G.renderheatmap(hm[1], transformedTips, gaussSD)
                    hm[2] = _G.renderheatmap(hm[2], transformedSpkt, gaussSD * 0.7)

                    if not inputbatch then
                        imageSize = input:size():totable()
                        inputbatch = torch.Tensor(sz, table.unpack(imageSize))
                    end
                    
                    if not heatmapbatch then
                        heatmapbatch = torch.zeros(sz, opt.nOutChannels, outputDim, outputDim)
                    end

                    if not targettable then
                        targettable = {}
                    end

                    if not normlengths then
                        normlengths = torch.zeros(sz)
                    end

                    if not batchawns then
                        batchawns = torch.zeros(sz,1,1,1)
                    end

                    inputbatch[i]:copy(input)
                    heatmapbatch[i]:copy(hm)
                    normlengths[i] = earlength
                    batchawns[i][1][1] = awns
                    table.insert(targettable, {tipsGTPoints, spktGTPoints})
                end

                collectgarbage()
                return {
                    input = inputbatch,
                    heatmap = heatmapbatch,
                    target = targettable,
                    indices = indices,
                    norms = normlengths,
                    awns = batchawns
                }
            end,
            function(_sample_)
                 sample = _sample_
            end,
            indices, self.split == 'train', opt
         )
         idx = idx + batchSize
        end
    end

    local n = 0
    local function loop()
        enqueue()
        if not threads:hasjob() then
         return nil
        end
        threads:dojob()
        if threads:haserror() then
         threads:synchronize()
        end
        enqueue()
        n = n + 1
        return n, sample
    end

    return loop
end

return M.DataLoader
