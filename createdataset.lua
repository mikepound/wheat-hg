require 'paths'
require 'torch'
require 'hdf5'
require 'string'
require 'image'
require 'json'

torch.setdefaulttensortype('torch.FloatTensor')

-- Project directory
rootDir = paths.concat(os.getenv('HOME'),'wheat')
imgDir = paths.concat(rootDir,'datasets/wheat')
genDir = paths.concat(rootDir,'gen')

function jpgsearch(file) 
    return file:find('.jpg') 
end

function file_exists(name)
    local f=io.open(name,"r")
    if f~=nil then io.close(f) return true else return false end
end

function in_bounds(pt, rect)
    return pt[1] >= rect[1] and pt[2] >= rect[2]
    and pt[1] < rect[1] + rect[3] and pt[2] < rect[2] + rect[4]
end

function scanfile(f)
    local function tabletonumber(tbl) 
        for i = 1, #tbl do
            tbl[i] = tonumber(tbl[i])
        end
    end

    -- Find and load json
    jsonName = string.gsub(f, ".jpg", ".json")
    jsonPath = paths.concat(imgDir, jsonName)
    imagePath = paths.concat(imgDir, f)

    img = image.load(imagePath);
    imgwidth = img:size(3)
    imgheight = img:size(2)

    if file_exists(jsonPath) then
        local o = json.load(jsonPath)
        local metadata = o["metadata"]
        local data = o["data"]
        
        if (metadata.Ignore == true) then
            return nil
        end

        ears = {}
        tips = {}
        spikelets = {}
    
        for k, v in pairs(data) do
            t = v["Type"]
            n = v["Name"]
            if t == "PolyLine" and n == "Ear" then
                -- Load points and convert to float tensor
                p = v["Points"]
                l = #p

                currentear = {};

                for i = 1,l do
                    row = string.split(p[i],",")
                    tabletonumber(row)
                    table.insert(currentear, row)
                end

                end1 = currentear[1]
                end2 = currentear[#currentear]

                if end1[2] < end2[2]
                then 
                    table.insert(tips, end1)
                else
                    table.insert(tips, end2)
                end

                table.insert(ears, torch.Tensor(currentear))

            elseif t == "Point" and n == "Spikelet" then
                row = string.split(v["Points"][1],",")
                tabletonumber(row)
                table.insert(spikelets, row)
            end
        end

        if #ears == 0 then
            return nil
        end

        return {
            ears = ears,
            tips = torch.Tensor(tips),
            spikelets = torch.Tensor(spikelets)
        }
    
    end

    return nil
end

padding = 30
capturedim = 384
outputdim = 256
sigma = 1.0
outputmaskdim = 64
maskscale = outputmaskdim / capturedim;

allfiles = {}
fileidx = 1
-- All JPEG files
for f in paths.files(imgDir, jpgsearch) do
    allfiles[fileidx] = f;
    fileidx = fileidx + 1
end

imagecount = #allfiles
traincount = torch.round(imagecount * 0.8)
validcount = imagecount - traincount
shuffle = torch.randperm(#allfiles)

trainidx = shuffle[{{1,traincount}}]
valididx = shuffle[{{traincount+1,-1}}]

trainingdata = {}

-- All training files
print ("Scanning training data...")
for idx = 1,trainidx:numel() do
    local f = allfiles[trainidx[idx]]
    local data = scanfile(f)
    if (data ~= nil) then
        table.insert( trainingdata, { filename = f, annotation = data } )
    end
end

print ("Training data size: ", #trainingdata)

validationdata = {}

-- All validation files
print ("Scanning validation data...")
for idx = 1,valididx:numel() do
    local f = allfiles[valididx[idx]]
    local data = scanfile(f)
    if (data ~= nil) then
        table.insert( validationdata, { filename = f, annotation = data } )
    end
end

print ("Validation data size: ", #validationdata)


print ("Saving data...")

data = {}
data['train'] = trainingdata;
data['valid'] = validationdata;

torch.save(paths.concat(genDir,'ears.t7'), data)

--for f in paths.files(imgDir, jpgsearch) do
--    processfile(f)

