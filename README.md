# wheat-hg

This is code that performs simultaneous classification, and localisation of features in wheat images. This is the code associated with our paper at the CVPPP workshop at ICCV2017:

[Deep Learning for Multi-Task Plant Phenotyping](http://openaccess.thecvf.com/content_ICCV_2017_workshops/w29/html/Pound_Deep_Learning_for_ICCV_2017_paper.html)

This code uses an adapted stacked hourglass network, as described [here](https://link.springer.com/chapter/10.1007/978-3-319-46484-8_29). We have altered the network architecture to add a classification branch, and the remaining code to support plant datasets such as our ACID dataset.

# Installation

Download the code by cloning the repository:

`git clone https://github.com/mikepound/wheat-hg.git`

The code requires that you have Torch installed, along with the lua-term and json libraries. Install torch, the install the dependencies with:

`luarocks install json`
`luarocks install lua-term`

# Install the ACID dataset

Next, download the ACID dataset and copy the images and json annotations into datasets/wheat/. Running the `createdataset.lua` and `updatedataset.lua` files will create a file ears.t7 within the gen folder, which stores lists of images in a training and validation set.

# Run the training
Finally, if these files are in place, you can train the network with a command like the following:

`th main.lua -GPU 1 -nThreads 4 -captureRes 512 -validate 3 -snapshot 100 -nEpochs 500 -LRStep 100 -LRStepGamma 0.5 -directory data-output-folder`

Details of all the command line parameters can be found in the opts.lua file, but here we are capturing input images crops at 512x512px resolution, validating every 3 epochs, and saving our trained model into the output directory every 100 epochs. In total we will run for 500 epochs, with the learning rate decreaseing by 1/2 every 100 epochs.

# Adapting to you own dataset

This code uses a common multi-thread torch dataloader to load images in, perform augmentations, and supply them to the network. If you wish to run a network like this on your own data, you will need to adapt dataloader, and ./datasets/ears.lua to target your own images and annotations.

If you have any questions about this code, feel free to contact me by email.