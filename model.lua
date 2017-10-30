--- Load up network model or initialize from scratch
paths.dofile('models/' .. opt.netType .. '.lua')

print('==> Creating model from file: models/' .. opt.netType .. '.lua')
model = createModel(modelArgs)

if opt.GPU ~= -1 then
    -- Convert model to CUDA
    print('==> Converting model to CUDA')
    model:cuda()
    
    cudnn.fastest = true
    cudnn.benchmark = true
end
