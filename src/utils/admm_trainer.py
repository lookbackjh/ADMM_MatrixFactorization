
import torch.nn as nn
import torch
class ADMM_MF_Block_Trainer():

    # to think :일단 작게 만들기? 
    # partitioning을 어떻게 할 것인가? , 업데이트를 어떻게 분리해서 하지 ,distributed 라는걸 어떻게 보여주지 
    def __init__(self,args,model,p_info) -> None:
        self.args = args
        self.model = model

        self.p_info=p_info  #  where p_info contains the data of the partition of the data
        self.tau_t=1

        pass


    def U_and_V_update(self,v_bar,up,vp,ratings,user_index,_item_index): 
        # for data in self.p_info:
        lambda_1=self.args.lambda_1
        up[user_index,:]=up[user_index,:]+self.tau_t*(self.get_epsilon(ratings,up,vp,user_index,_item_index)-self.args.lambda_1*up[user_index,:])

        pass

    def get_epsilon(self,ratings,up,vp,user_index,_item_index):
        to_return=ratings[user_index,_item_index]-torch.mul(up[user_index,:],vp[_item_index,:]) # np로 바꿔야함
        return to_return
