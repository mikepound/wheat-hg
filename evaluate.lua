require 'torch'
require 'image'

local M = {}

local function getdistances(pt, gt)
    d = torch.zeros(pt:size(1))
    for i = 1,d:size(1) do
        local min = 100.0
        for j = 1,gt:size(1) do
            dist = torch.dist(pt[i],gt[j])
            if dist < min then min = dist end
        end
        d[i] = min
    end
    return d
end

local function nonmaximalsuppression(pred, threshold)
    
    local points = {};
    local function addpoint(i, stride, points)
        local x = i % stride
        local y = math.floor(i / stride)
        -- Add +1/+1 because data() access is zero, not one indexed
        table.insert(points, {x+1,y+1})
    end

    -- This is now zero indexed data
    local pred_data = pred:contiguous():data() -- Much faster than normal tensor indexing
    local stride = pred:stride(2) -- Should always be 64
    local numel = pred:numel()

    -- Top
    for i = 0,stride-1 do
        local val = pred_data[i]
        if val < threshold then
            pred_data[i] = 0
        elseif  val >= pred_data[i-1]
            and val >= pred_data[i+1]
            and val >= pred_data[i+stride] then
            addpoint(i, stride, points)
        end
    end

    -- Bottom
    for i = numel-stride,numel-1 do
        local val = pred_data[i]
        if val < threshold then
            pred_data[i] = 0
        elseif  val >= pred_data[i-1]
            and val >= pred_data[i+1]
            and val >= pred_data[i-stride] then
            addpoint(i, stride, points)
        end
    end

    -- Other rows including left and right columns
    local rowoffset = stride
    local term = numel - stride
    while rowoffset < term do
        local rowend = rowoffset+stride-1
        for i = rowoffset,rowend do
            local val = pred_data[i]
            if val < threshold then
                pred_data[i] = 0
            elseif i == rowoffset then
                if  val >= pred_data[i+1]
                and val >= pred_data[i+stride]
                and val >= pred_data[i-stride] then
                    addpoint(i, stride, points)
                end
            elseif i < rowend then
                if  val >= pred_data[i-1]
                and val >= pred_data[i+1]
                and val >= pred_data[i+stride]
                and val >= pred_data[i-stride] then
                    addpoint(i, stride, points)
                end
            else
                if  val >= pred_data[i-1]
                and val >= pred_data[i+stride]
                and val >= pred_data[i-stride] then
                    addpoint(i, stride, points)
                end
            end
        end
        rowoffset = rowoffset + stride
    end

    return torch.Tensor(points)
end

local function evaluatesinglechannel(prhm, gtpoints, nmsthreshold, distancethreshold)
    local prpoints = nonmaximalsuppression(prhm, nmsthreshold)

    if prpoints:numel() == 0 or gtpoints:numel() == 0 then
        -- Empty tensor, either early in the training process, or an empty image
        if prpoints:numel() == gtpoints:numel() then
            -- Both no predicted points and no target points
            return { tp = 0, fp = 0, fn = 0 }
        elseif prpoints:numel() == 0 then
            -- No predicted points, all false negatives
            return { tp = 0, fp = 0, fn = gtpoints:numel() }
        else
            -- No ground truth points, all false positives
            return { tp = 0, fp = prpoints:numel(), fn = 0  }
        end
    end

    local prdist = getdistances(prpoints, gtpoints)
    local gtdist = getdistances(gtpoints, prpoints)

    local prresult = prdist:le(distancethreshold)
    local gtresult = gtdist:le(distancethreshold)

    local tp = prresult:sum()
    local fp = (1-prresult):sum()
    local fn = (1-gtresult):sum()

    return { tp = tp, fp = fp, fn = fn }
end

local function evaluatesinglechannelbatch(prhm, gtpoints, nmsthreshold, distancethreshold)
    local batchtp, batchfp, batchfn = 0, 0, 0
    assert(prhm:size(1) == #gtpoints, "Heatmap tensor does not match ground truth table size" .. prhm:size(1) .. " " .. #gtpoints)

    for i = 1,prhm:size(1) do
        local single = evaluatesinglechannel(prhm[i], gtpoints[i], nmsthreshold, distancethreshold)
        batchtp = batchtp + single.tp
        batchfp = batchfp + single.fp
        batchfn = batchfn + single.fn
    end

    return {
        tp = batchtp,
        fp = batchfp,
        fn = batchfn
    }

end

local function evaluatemultichannelbatch(prhm, gtpoints, nmsthreshold, normdistancethresholds)
    local batchcount = prhm:size(1)
    local channelcount = prhm:size(2)

    --local batchtp, batchfp, batchfn = 0, 0, 0, 0
    --assert(prhm:size(1) == #gtpoints, "Heatmap tensor does not match ground truth table size" .. prhm:size(1) .. " " .. #gtpoints)
    channelresults = {}
    for channel = 1,channelcount do
        local ctp, cfp, cfn = 0, 0, 0
        for batch = 1,batchcount do
            assert(#gtpoints[batch] == channelcount, "Number of ground truth channels does not match heatmap channels")
            
            local normthreshold = normdistancethresholds[channel][batch]

            local single = evaluatesinglechannel(prhm[batch][{{channel}, {},{}}], gtpoints[batch][channel], nmsthreshold, normthreshold)
            ctp = ctp + single.tp
            cfp = cfp + single.fp
            cfn = cfn + single.fn
        end
        table.insert(channelresults, { tp = ctp, fp = cfp, fn = cfn })

    end

    return channelresults

end

local function calculateF1(tp, fp, fn)
    local precision, recall = 0.0, 0.0
    
    if tp + fp > 0 then precision = tp / (tp + fp) end
    if tp + fn > 0 then recall = tp / (tp + fn) end

    local f1 = 0.0
    if precision + recall > 0 then f1 = 2 * (precision * recall) / (precision + recall) end
    return precision, recall, f1
end

local function calculateMultiF1(score)
    assert(type(score) == 'table', "Score is not a table")

    local prf1 = {}

    for i = 1,#score do
        local p,r,f1
        s = score[i]
        p,r,f1 = calculateF1(s.tp, s.fp, s.fn)
        table.insert(prf1, { precision = p, recall = r, f1 = f1 })
    end

    return prf1
end

local function evaluateclassification(pred, gt)
    assert(pred:numel() == gt:numel(), "Class prediction and scores are different sizes")
    local npos = pred:sum()
    local correct = torch.eq(pred, gt)
    local tp = torch.cmul(pred, correct):sum()
    local fn = torch.lt(pred,gt):sum()

    return {
        tp = tp,
        fp = npos - tp,
        tn = pred:size(1) - npos - fn,
        fn = fn
    }
end

M.nonmaximalsuppression = nonmaximalsuppression
M.getdistances = getdistances
M.evaluatesinglechannel = evaluatesinglechannel
M.evaluatesinglechannelbatch = evaluatesinglechannelbatch
M.calculateF1 = calculateF1
M.calculateMultiF1 = calculateMultiF1
M.evaluatemultichannelbatch = evaluatemultichannelbatch
M.evaluateclassification = evaluateclassification
return M