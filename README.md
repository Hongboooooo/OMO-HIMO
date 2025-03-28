# OMO-HIMO
Modify and Retrain omo[1] 1st stage network for multiple objects manipulation generation using himo dataset[2]
> Task Description: given mutliple objects geometries and motion sequences, predict the position sequence of two hands  
> Current Progress:  
>   1. Built up the data processing functions
>   2. Built up the diffusion network's training and evaluation framework  
>   3. Modified architecture by explanding the object encoder channel

> Next Step:
>   1. Introduce a contact guidance to avoid hand position sliding when grasping objects
>>    similiar with classifier guidance, but instead of using classifier during training stage, going to design an energy function to guide the gradient of denosing during inference

Visual Result:
> Two green dots in the gif below are the inferred result of network
![image](https://github.com/Hongboooooo/OMO-HIMO/blob/main/omo-himo.gif)


Reference:
[1] https://github.com/lijiaman/omomo_release  
[2] 
