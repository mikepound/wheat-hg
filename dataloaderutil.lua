local M = {}

local cache = {}

local function drawGaussian(img, pt, sigma)
	local height,width = img:size(1),img:size(2)

	-- Draw a 2D gaussian
	-- Check that any part of the gaussian is in-bounds
	local tmpSize = math.ceil(3*sigma)

	local ul = {math.floor(pt[1] - tmpSize), math.floor(pt[2] - tmpSize)}
	local br = {math.floor(pt[1] + tmpSize), math.floor(pt[2] + tmpSize)}

	-- If not, return the image as is
	if (ul[1] > width or ul[2] > height or br[1] < 1 or br[2] < 1) then return img end
	-- Generate gaussian
	local size = 2*tmpSize + 1

	if not cache[size] then
		cache[size] = image.gaussian(size)
	end

	local g = cache[size]

	-- Usable gaussian range
	local g_x = {math.max(1, 2-ul[1]), math.min(size, size + (width - br[1]))}
	local g_y = {math.max(1, 2-ul[2]), math.min(size, size + (height - br[2]))}

	-- Image range
	local img_x = {math.max(1, ul[1]), math.min(br[1], width)}
	local img_y = {math.max(1, ul[2]), math.min(br[2], height)}
	
	img:sub(img_y[1], img_y[2], img_x[1], img_x[2]):cmax(g:sub(g_y[1], g_y[2], g_x[1], g_x[2]))
	return img
end

local function renderheatmap(img, pts, sd)
	for g = 1, pts:size(1) do
		img = drawGaussian(img, pts[g], sd)
	end
	return img
end

M.renderheatmap = renderheatmap
M.drawGaussian = drawGaussian
return M