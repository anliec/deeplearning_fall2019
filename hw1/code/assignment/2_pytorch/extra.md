# Open challenge methodology

For this section I used the code and pre-trained weights from https://github.com/quark0/darts
I reimplemented the authors network on the homework format, modified the `train.py` script to 
allow fine tuning of existing weights and added data augmentation on the training data. The 
pre-trained model being already trained on Cifar-10 using architecture search, no much
training were needed, one epoch was enough for the network to take into account the 
difference on dataset loading.

I expect one or two percent accuracy can still be gained by re running the architecture
search and using more fine tuned data augmentation.

- Name: Nicolas Six
- Email ID: nsix6 (nsix6@gatech.edu)
- Best accuracy: 0.962 (EvalAI public set)
