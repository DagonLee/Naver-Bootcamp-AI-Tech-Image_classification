#!/usr/bin/env python
# coding: utf-8

# In[3]:


"""

## Pytroch

## Requriements : torchviz, pytorch

## print submodule and weight when call method backward
## if grad is None print None
## if grad is nan , print nan
## then print grad meand

#output: module name

#     :  grad_out : grad to next layer

#    :  grad_in :  grad to previous layer
     
"""
    
## prerequisite : you must know about  backward process                                    
def output_gradient_module(model):
    def checknan(grad):
    #     print(grad)
        if grad is None : return None
        if  torch.any(grad.isnan()) : return 'Nan'
        else : return torch.mean(grad)
    def backward_hook(module,grad_in,grad_out):
        print('module' , module)
        print('grad_input', list(map(checknan,grad_in)))
        print('grad_out',list(map(checknan,grad_out)))
        print('\n\n')
        print('------------------------')
    for name, md in model.named_modules():
        md.register_full_backward_hook(backward_hook)
        

"""
    ploting backward graph

"""

def plot_backward_model(model):
    return make_dot(ls,dict(list(model.named_parameters())) , show_attrs=True ,show_saved=True)


# In[4]:





# In[ ]:




