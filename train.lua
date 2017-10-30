-- Main training function

function step(tag)
    local avgLoss, avgAcc = 0.0, 0.0
    local avgLoss = 0.0
    local score = {}
    local classScore = { tp = 0, fp = 0, tn = 0, fn = 0 }
    local output, err, idx
    local param, gradparam = model:getParameters()
    local function evalFn(x) return criterion.output, gradparam end
    local images, labels, batch

    if tag == 'train' then
        model:training()
        set = 'train'
        loader = trainLoader
    else
        model:evaluate()
        set = 'valid'
        loader = validLoader
    end

    local nIters,datasetsize,batchsize = loader:size(), loader:datasetsize(), loader:batchsize()

    for i,sample in loader:run() do
        local input = sample.input;
        local masks = sample.heatmap;
        local target = sample.target;
        local idx = sample.indices;
        local norms = sample.norms
        local awns = sample.awns:byte()

        xlua.progress(i, nIters)
        
        if opt.GPU ~= -1 then
            -- Convert to CUDA
            input = input:cuda()
            masks = masks:cuda()
            class = awns:cuda()
        end

        -- Multiple supervision table
        local label;
        if opt.nStack > 1 then
            label = {}
            for t = 1,4 do
                label[t] = masks
            end
            table.insert(label, class)
        else
            label = {masks, class}
        end

        -- Do a forward pass and calculate loss
        local output = model:forward(input)
        --local o = {}
        --for j = 1,4 do o[j] = output[j] end
        local err = criterion:forward(output, label)

        avgLoss = avgLoss + err / nIters

        if tag == 'train' then
            -- Training: Do backpropagation and optimization
            model:zeroGradParameters()
            model:backward(input, criterion:backward(output, label))
            optfn(evalFn, param, optimState)
        end

        -- Calculate normalised distances or F1
        local normalisedchannelthresholds = { 0.2, 0.1}
        normthresholds = torch.Tensor(#normalisedchannelthresholds,norms:size(1))

        for n = 1, #normalisedchannelthresholds do
           normthresholds[n] = torch.mul(norms,normalisedchannelthresholds[n])
        end

        -- Calculate accuracy
        local evaluation = eval.evaluatemultichannelbatch(output[opt.nStack]:float(), target, 0.5, normthresholds)
        if #score == 0 then
            for s = 1,#evaluation do
                table.insert(score, { tp = 0, fp = 0, fn = 0 })
            end
        end

        for s = 1,#evaluation do
            score[s].tp = score[s].tp + evaluation[s].tp
            score[s].fp = score[s].fp + evaluation[s].fp
            score[s].fn = score[s].fn + evaluation[s].fn
        end

        -- Classification Accuracy
        local classpred = torch.gt(output[opt.nStack+1]:float(), 0.5) 
        local classevaluation = eval.evaluateclassification(classpred, awns)

        classScore.tp = classScore.tp + classevaluation.tp
        classScore.fp = classScore.fp + classevaluation.fp
        classScore.tn = classScore.tn + classevaluation.tn
        classScore.fn = classScore.fn + classevaluation.fn

        collectgarbage();
    end

    return avgLoss, score, classScore

end

function train() return step('train') end
function valid() return step('valid') end
