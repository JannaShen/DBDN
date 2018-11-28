local paths = require 'paths'
local transform = require 'data/transforms'

local M = {}
local div2k = torch.class('sr.div2k', M)

function div2k:__init(opt, split)
    self.opt = opt
    self.split = split

    self.nTrain = opt.nTrain_DIV2K
    self.nVal = opt.nVal
	self.offset = opt.valOffset -- (offset + 1) ~ (offset + nVal) images are used to validate the training
    self.scale = self.opt.scale
    self.dataSize = self.opt.dataSize

    --Absolute path of the dataset
    local apath = nil
    self.ext = nil

    if opt.datatype == 'png' then
        apath = paths.concat(opt.datadir, 'DIV2K')
        self.ext = '.png'
    elseif opt.datatype == 't7' or opt.datatype == 't7pack' then
        apath = paths.concat(opt.datadir, 'DIV2K_decoded')
        self.ext = '.t7'
    else
        error('unknown -datatype (png | t7(default) | t7pack)')
    end

    local tHR = 'DIV2K_train_HR'
    local tLR = 'DIV2K_train_LR_'

    self.dirTar = paths.concat(apath, tHR)
    self.dirInp, self.dirInp_aug = {}, {}

    for i = 1, #self.scale do
		table.insert(self.dirInp, paths.concat(apath, tLR .. opt.degrade, 'X' .. self.scale[i]))
		if opt.augUnkDIV2K then
            table.insert(self.dirInp_aug, paths.concat(apath, tLR .. opt.degrade .. '_augment', 'X' .. self.scale[i]))
		end
        self.dirInp[i] = opt.dataSize == 'small' and self.dirInp[i] or self.dirInp[i]
        self.dirInp[i] = opt.netType ~= 'recurVDSR' and self.dirInp[i] or self.dirInp[i] .. '_SRresOutput'
    end

    --Load single .t7 files that contains all dataset
    if opt.datatype == 't7pack' then
        assert(not opt.augUnkDIV2K, 'Cannot use t7pack if you select -augUnkDIV2K true')
        print('\tLoading t7pack:')
        if split == 'train' then
            --Here, we will split the validation sets and save them as *v.t7 file
            self.t7Tar = torch.load(paths.concat(self.dirTar, 'pack.t7'))
            local valImgs = {table.unpack(self.t7Tar, self.offset + 1, self.offset + self.nVal)}
            torch.save(paths.concat(self.dirTar, 'pack_v.t7'), valImgs)
            print('\tTrain set: ' .. self.dirTar .. '/pack.t7 loaded')

            self.t7Inp = {}
            for i = 1, #self.dirInp do
                if self.scale[i] ~= 1 then
                    table.insert(self.t7Inp, torch.load(paths.concat(self.dirInp[i], 'pack.t7')))
                    local valImgs = {table.unpack(self.t7Inp[i], self.offset + 1, self.offset + self.nVal)}
                    torch.save(paths.concat(self.dirInp[i], 'pack_v.t7'), valImgs)
                    print('\tTrain set: ' .. self.dirInp[i] .. '/pack.t7 loaded')
                else
                    table.insert(self.t7Inp, self.t7Tar)
                end
            end
        elseif split == 'val' then
            self.t7Tar = torch.load(paths.concat(self.dirTar, 'pack_v.t7'))
            print('\tValidation set: ' .. self.dirTar .. '/pack_v.t7 loaded')
            self.t7Inp = {}
            for i = 1, #self.dirInp do
                if self.scale[i] ~= 1 then
                    table.insert(self.t7Inp, torch.load(paths.concat(self.dirInp[i], 'pack_v.t7')))
                    print('\tValidation set: ' .. self.dirInp[i] .. '/pack_v.t7 loaded')
                else
                    table.insert(self.t7Inp, self.t7Tar)
                end
            end
        end
    end

    collectgarbage()
    collectgarbage()
end

function div2k:get(idx, scaleIdx)
    local idx = idx
    local scale = self.scale[scaleIdx]
    local dataSize = self.dataSize

    if self.split == 'train' then
        if idx > self.offset then
            idx = idx + self.nVal
        end
    elseif self.split == 'val' then
        if self.opt.datatype ~= 't7pack' then
            idx = idx + self.offset
        end
    end

    local input, target
    local inputName, targetName, rot
    if self.opt.datatype == 't7pack' then
        if  self.opt.augRF then
            rot = math.random(1, 8)
        else
            rot = 1
        end
        
        input = self.t7Inp[scaleIdx][idx]
        target = self.t7Tar[idx]
    elseif self.opt.datatype == 't7' then
        inputName, targetName, rot = self:getFileName(idx, scale)
		if self.split == 'train' and self.opt.augUnkDIV2K then
			input = torch.load(paths.concat(self.dirInp_aug[scaleIdx], inputName))
		else
			input = torch.load(paths.concat(self.dirInp[scaleIdx], inputName))
		end

        target = torch.load(paths.concat(self.dirTar, targetName))
    elseif self.opt.datatype == 'png' then
        inputName, targetName, rot = self:getFileName(idx, scale)
		if self.split == 'train' and self.opt.augUnkDIV2K then
			input = image.load(paths.concat(self.dirInp_aug[scaleIdx], inputName), self.opt.nChannel, 'float')
		else
			input = image.load(paths.concat(self.dirInp[scaleIdx], inputName), self.opt.nChannel, 'float')
		end
        target = image.load(paths.concat(self.dirTar, targetName), self.opt.nChannel, 'float')
    end
    
    local _, h, w = unpack(target:size():totable())
    local hInput, wInput = math.floor(h / scale), math.floor(w / scale)
    local hTarget, wTarget = scale * hInput, scale * wInput
    target = target[{{}, {1, hTarget}, {1, wTarget}}]
    
    local patchSize = self.opt.patchSize
    local targetPatch = self.opt.multiPatch and (patchSize * scale) or patchSize
    local inputPatch = (self.opt.dataSize == 'big') and targetPatch or targetPatch / scale

    if (wTarget < targetPatch) or (hTarget < targetPatch) then
        return nil
    end

    --Generate patches for training
    if self.split == 'train' then
        local ix = torch.random(1, wInput - inputPatch + 1)
        local iy = torch.random(1, hInput - inputPatch + 1)
        local tx = scale * (ix - 1) + 1
        local ty = scale * (iy - 1) + 1
        input = input[{{}, {iy, iy + inputPatch - 1}, {ix, ix + inputPatch - 1}}]
        target = target[{{}, {ty, ty + targetPatch - 1}, {tx, tx + targetPatch - 1}}]
        -- data augmentation on patches
        --rot = math.random(1,8)
        --we always conduct data augmentation
        if self.opt.augRF then
           rot = math.random(1,8)
           --print(rot)
           input, target = self:dataAugmentation(input, target, rot)
        end
        
    end

    if self.opt.datatype == 'png' then
        input:mul(self.opt.mulImg)
        target:mul(self.opt.mulImg)
    else
        input = input:float():mul(self.opt.mulImg / 255)
        target = target:float():mul(self.opt.mulImg / 255)
    end

    --Reject the patch that has small size of spatial gradient
    if self.split == 'train' and self.opt.rejection ~= -1 then
        local grT, grP = nil, nil
        if self.opt.rejectionType == 'input' then
            grT, grP = input, inputPatch
        elseif self.opt.rejectionType == 'target' then
            grT, grP = target, targetPatch
        end

        local dx = grT[{{}, {1, grP - 1}, {1, grP - 1}}] - grT[{{}, {1, grP - 1}, {2, grP}}]
        local dy = grT[{{}, {1, grP - 1}, {1, grP - 1}}] - grT[{{}, {2, grP}, {1, grP - 1}}]
        local dsum = dx:pow(2) + dy:pow(2)
        local dsqrt = dsum:sqrt()
        local gradValue = dsqrt:view(-1):mean()
        
        if self.gradStatistics == nil then
            self.gradSamples = self.opt.nGradStat
            self.gsTable = {}
            self.gradStatistics = {}
            for i = 1, #self.scale do
                table.insert(self.gsTable, {})
                table.insert(self.gradStatistics, -1)
            end
            print('Caculating median of gradient for ' .. self.gradSamples .. ' samples...')
            return nil
        end
        
        if #self.gsTable[scaleIdx] < self.gradSamples then
            table.insert(self.gsTable[scaleIdx], gradValue)
            return nil
        else
            if self.gradStatistics[scaleIdx] == -1 then
                local threshold = math.floor(self.gradSamples * self.opt.rejection / 100)
                table.sort(self.gsTable[scaleIdx])
                self.gradStatistics[scaleIdx] = self.gsTable[scaleIdx][threshold]
                print('Gradient threshold for scale ' .. self.scale[scaleIdx] .. ': ' .. self.gradStatistics[scaleIdx])
                return nil
            else
                if gradValue <= self.gradStatistics[scaleIdx] then
                    return nil
                end
            end
        end
    end

    return {
        input = input,
        target = target
    }
end

function div2k:__size()
    if self.split == 'train' then
        return self.nTrain
    elseif self.split == 'val' then
        return self.nVal
    end
end

function div2k:augment()
    if self.split == 'train' and self.opt.degrade == 'bicubic' then
        local transforms = {}
        if self.opt.colorAug then
            table.insert(transforms,
                transform.ColorJitter({
                    brightness = 0.1,
                    contrast = 0.1,
                    saturation = 0.1
                })
            )
        end
        -- We don't need vertical flip, since hflip + rotation covers it
        table.insert(transforms, transform.HorizontalFlip())
        table.insert(transforms, transform.Rotation())

        return transform.Compose(transforms)
    else
        return function(sample) return sample end
    end
end
-- data augmentation, 

function div2k:dataAugmentation(input, target, rot)
    if rot == 1 then        -- original 
       input  = input
       target = target
    elseif rot == 2 then    -- rotation 90
       --print('rot=2')
       input  = image.vflip(input)
       target = image.vflip(target)
    elseif rot == 3 then    -- rotation 180
       input  = image.hflip(input)
       target = image.hflip(target)
    elseif rot == 4 then    -- rotation 270
       input  = image.hflip(image.vflip(input))
       target = image.hflip(image.vflip(target))
    elseif rot == 5 then    -- original & flip(vertical)
       input  = image.rotate(input, math.pi/2)
       target = image.rotate(target, math.pi/2)
    elseif rot == 6 then    -- rotation 90 & flip(vertical)
       --print('rot=6')
       input  = image.vflip(image.rotate(input, math.pi/2))
       target = image.vflip(image.rotate(target, math.pi/2))
    elseif rot == 7 then    -- rotation 180 & flip(vertical)
       input  = image.hflip(image.rotate(input, math.pi/2))
       target = image.hflip(image.rotate(target, math.pi/2))
    elseif rot == 8 then    -- rotation 270 & flip(vertical)
       input  = image.hflip(image.vflip(image.rotate(input, math.pi/2)))
       target = image.hflip(image.vflip(image.rotate(target, math.pi/2)))
    end
    return input, target
end


function div2k:getFileName(idx, scale)
    --filename format: ????x?.png
    local fileName = idx
    local digit = idx
    while digit < 1000 do
        fileName = '0' .. fileName
        digit = digit * 10
    end

    local targetName = fileName .. self.ext
    local inputName = nil
    local rot
    local BD
    if scale == 1 then
        inputName = targetName
    else
        if self.opt.netType == 'recurVDSR' then
            inputName = 'SRres' .. fileName .. 'x' .. scale .. self.ext
        else
            if self.split == 'train' and self.opt.augUnkDIV2K then
                rot = math.random(1,8)
                inputName = fileName .. 'x' .. scale .. '_' .. rot .. self.ext
            elseif self.split == 'train' then
                rot = nil
                BD = math.random(1,2)
                if self.opt.DownKernel == 'BD' then
                    --print('train BD')
                    inputName = fileName .. 'x' .. scale .. 'BD' .. self.ext
                elseif self.opt.DownKernel == 'DN' then
                    --print('train DN')
                    inputName = fileName .. 'x' .. scale .. 'N3' .. self.ext
                elseif self.opt.DownKernel == 'BI' then
                    --print('train BI')
                    inputName = fileName .. 'x' .. scale .. self.ext
                elseif self.opt.DownKernel == 'BIBD' then
                    if BD == 1 then
                        inputName = fileName .. 'x' .. scale .. 'BD' .. self.ext
                    else
                        inputName = fileName .. 'x' .. scale .. self.ext
                    end
                end
                
                --print(rot)
            elseif self.split == 'val' and self.opt.DownKernel == 'BD' then
                --print('val BD')
                rot = nil
                inputName = fileName .. 'x' .. scale .. 'BD' .. self.ext
            elseif self.split == 'val' and self.opt.DownKernel == 'DN' then
                --print('val BD')
                rot = nil
                inputName = fileName .. 'x' .. scale .. 'N3' .. self.ext
            else
                rot = nil
                inputName = fileName .. 'x' .. scale .. self.ext
            end
        end
    end

    return inputName, targetName, rot
end

return M.div2k
