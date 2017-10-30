require 'torch'
require 'xlua'
require 'optim'
require 'nn'
require 'nnx'
require 'nngraph'
require 'hdf5'
require 'string'
require 'image'
require 'paths'

ffi = require 'ffi'
eval = require('evaluate');

local term = require 'term'
local colors = term.colors
function writeformatted(set, loss, f1, classp, color)
	local f1string
	if type(f1) == "table" then
		f1string = string.format("%.7f" % {f1[1].f1})
		for i = 2,#f1 do
			f1string = string.format(f1string .. ",%.7f" % f1[i].f1)
		end
	else
		f1string = string.format("%.7f" % f1)
	end
	print(string.format("      %s%s%s : Loss: %.7f, F1: %s%s%s, Class acc.: %s%0.2f%s" % { color, set, colors.reset, loss, color, f1string, colors.reset, color, classp, colors.reset}))
end

torch.setdefaulttensortype('torch.FloatTensor')

-- Options
paths.dofile('opts.lua')
opt.nOutChannels = 2

-- Output directory
local workingDir = './snapshots/'
if opt.directory ~= '' then
	workingDir = paths.concat(workingDir, opt.directory)
end

-- Data loading
local DataLoader = require 'dataloader'
trainLoader, validLoader = DataLoader.create(opt)

-- Initialise Logging 
if not Logger then paths.dofile('Logger.lua') end

if opt.model == 'none' then
	--- Load up network model or initialize from scratch
	paths.dofile('models/' .. opt.netType .. '.lua')

	print('==> Creating model from file: models/' .. opt.netType .. '.lua')
	model = createModel(modelArgs)

else
	print ("==> Loading model from existing file:", opt.model)
	model = torch.load(opt.model)
end

if opt.GPU ~= -1 then
	-- Convert model to CUDA
	print('==> Converting model to CUDA')
	model:cuda()
	
	cudnn.fastest = true
	cudnn.benchmark = true
end

-- Criterion
criterion = nn.ParallelCriterion()
for i = 1,opt.nStack do
	criterion:add(nn[opt.crit .. 'Criterion'](), 1.0)
end
criterion:add(nn.BCECriterion(), 5e-2)
criterion:cuda()

if opt.optimState == 'none' then
	-- Optimisation
	print('==> Creating optimState from scratch')
	optfn = optim[opt.optMethod]
	if not optimState then
		optimState = {
			learningRate = opt.LR,
			learningRateDecay = opt.LRdecay,
			momentum = opt.momentum,
			weightDecay = opt.weightDecay,
			alpha = opt.alpha,
			epsilon = opt.epsilon
		}
	end
else
	print ("==> Loading optimState from existing file: ", opt.optimState)
	optimState = torch.load(opt.optimState)
	optfn = optim[opt.optMethod]
end

-- Load training code
paths.dofile('train.lua')

-- Train and validate for a while
local loss, f1 = 0.0, 0.0
local tp, fp, fn = 0, 0, 0
local scores, f1scores, classScores, classAcc

local epoch = 1
if optimState.epoch then
	epoch = optimState.epoch + 1
	print ("Resuming from optimState at epoch " .. epoch)
	
end

local continue = epoch > 1
log = {}
log.train = Logger(paths.concat(workingDir, 'train.log'), continue)
log.train:setNames{'epoch', 'loss', 'ears.tp', 'ears.fp' ,'ears.fn', 'spkt.tp', 'spkt.fp', 'spkt.fn', 'class.tp', 'class.fp', 'class.tn', 'class.fn', 'lr'}
log.valid = Logger(paths.concat(workingDir, 'valid.log'), continue)
log.valid:setNames{'epoch', 'loss', 'ears.tp', 'ears.fp' ,'ears.fn', 'spkt.tp', 'spkt.fp', 'spkt.fn', 'class.tp', 'class.fp', 'class.tn', 'class.fn', 'lr'}

print ("Working directory: " .. workingDir)

while epoch <= opt.nEpochs do
	print ("Epoch " .. epoch)

	loss, scores, classScores = train()

	f1scores = eval.calculateMultiF1(scores)

	local correct = (classScores.tp + classScores.tn)
	classAcc =  correct / (correct + classScores.fp + classScores.fn) 

	writeformatted('Train', loss, f1scores, classAcc, colors.green)

	-- Logging
	if log['train'] then
		log['train']:add{
			string.format("%d" % epoch),
			string.format("%.6f" % loss),
			
			string.format("%d" % scores[1].tp),
			string.format("%d" % scores[1].fp),
			string.format("%d" % scores[1].fn),

			string.format("%d" % scores[2].tp),
			string.format("%d" % scores[2].fp),
			string.format("%d" % scores[2].fn),

			string.format("%d" % classScores.tp),
			string.format("%d" % classScores.fp),
			string.format("%d" % classScores.tn),
			string.format("%d" % classScores.fn),

			string.format("%g" % optimState.learningRate)
		}
	end

	-- If we are validating this epoch
	if (opt.validate ~= 0 and epoch % opt.validate == 0) then
		
		local multiclassScore
		local multiscore

		for i = 1,opt.validateIterations do
			loss, scores, classScores = valid();

			if not multiscore then
				multiscore = scores
			else
				for s = 1,#multiscore do
					multiscore[s].tp = multiscore[s].tp + scores[s].tp
					multiscore[s].fp = multiscore[s].fp + scores[s].fp
					multiscore[s].fn = multiscore[s].fn + scores[s].fn
				end

			end

			if not multiclassScore then
				multiclassScore = classScores
			else
				multiclassScore.tp = multiclassScore.tp + classScores.tp
				multiclassScore.fp = multiclassScore.fp + classScores.fp
				multiclassScore.tn = multiclassScore.tn + classScores.tn
				multiclassScore.fn = multiclassScore.fn + classScores.fn
			end
		end

		f1scores = eval.calculateMultiF1(multiscore)

		local correct = (classScores.tp + classScores.tn)
		classAcc =  correct / (correct + classScores.fp + classScores.fn) 

		writeformatted('Valid', loss, f1scores, classAcc, colors.cyan)

		if log['valid'] then
			log['valid']:add{
				string.format("%d" % epoch),
				string.format("%.6f" % loss),
				
				string.format("%d" % multiscore[1].tp),
				string.format("%d" % multiscore[1].fp),
				string.format("%d" % multiscore[1].fn),

				string.format("%d" % multiscore[2].tp),
				string.format("%d" % multiscore[2].fp),
				string.format("%d" % multiscore[2].fn),

				string.format("%d" % multiclassScore.tp),
				string.format("%d" % multiclassScore.fp),
				string.format("%d" % multiclassScore.tn),
				string.format("%d" % multiclassScore.fn),

				string.format("%g" % optimState.learningRate)
			}
		end
	end

	optimState.epoch = epoch

	-- If snapshotting on this epoch
	if epoch % opt.snapshot == 0 then
		print ("Saving model")
		model:clearState()
		torch.save(paths.concat(workingDir, 'model_' .. epoch .. '.t7') , model)
		torch.save(paths.concat(workingDir, 'optimState_' .. epoch .. '.t7'), optimState)
	end

	if (epoch % opt.LRStep == 0) then
		optimState.learningRate = optimState.learningRate * opt.LRStepGamma
		print ("Reducing learning rate to " .. optimState.learningRate)
	end
	epoch = epoch + 1
end
