
class ADMM_MF_Trainer():

    def __init__(self,args,model) -> None:
        self.args = args
        self.model = model
        self.v_bar=self.model.item_embedding.weight
        pass


    def U_and_V_update(self) :
        up= self.model.user_embedding.weight
        vp= self.model.item_embedding.weight

        pass
