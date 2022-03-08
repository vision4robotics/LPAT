# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from pysot.core.config import cfg
from pysot.models.loss import select_cross_entropy_loss, INDLoss, weight_l1_loss,l1loss,IOULoss,gIOULoss
from pysot.models.backbone.newalexnet import AlexNet
from pysot.models.utile.utile import LOGO
import numpy as np



class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        self.backbone = AlexNet().cuda()
        self.grader=LOGO(cfg).cuda()



        self.cls3loss=nn.BCEWithLogitsLoss()
        self.IOULOSS=IOULoss()
        self.gIOULoss=gIOULoss()  
        self.INDLOSS=INDLoss()        
        
    def template(self, z):
        with t.no_grad():
            zf = self.backbone(z)
    
            self.zf=zf
            
       # self.zf1=zf1

    
    def track(self, x):
        with t.no_grad():
            
            xf = self.backbone(x)  
            loc,cls2,cls3=self.grader(xf,self.zf)

            return {

                'cls2': cls2,
                'cls3': cls3,
                'loc': loc
               }

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2//2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)

        return cls


    def getcenter(self,mapp):

        def con(x):
            return x*143
        
        size=mapp.size()[3]
        #location 
        x=np.tile((16*(np.linspace(0,size-1,size))+63)-287//2,size).reshape(-1)
        y=np.tile((16*(np.linspace(0,size-1,size))+63).reshape(-1,1)-287//2,size).reshape(-1)
        shap=con(mapp).cpu().detach().numpy()
        xx=np.int16(np.tile(np.linspace(0,size-1,size),size).reshape(-1))
        yy=np.int16(np.tile(np.linspace(0,size-1,size).reshape(-1,1),size).reshape(-1))

        # xx=xx.reshape(-1,1).repeat(repeats=cfg.TRAIN.BATCH_SIZE,axis=1)
        # yy=yy.reshape(-1,1).repeat(repeats=cfg.TRAIN.BATCH_SIZE,axis=1)

        # w=np.abs(shap[:,2,yy,xx]*143)
        # h=np.abs(shap[:,3,yy,xx]*143)
        # x=x+shap[:,0,yy,xx]*80
        # y=y+shap[:,1,yy,xx]*80
        w=shap[:,0,yy,xx]+shap[:,1,yy,xx]
        h=shap[:,2,yy,xx]+shap[:,3,yy,xx]
        x=x-shap[:,0,yy,xx]+w/2
        y=y-shap[:,2,yy,xx]+h/2
        # w=np.abs(shap[:,0,yy,xx]*143)
        # h=np.abs(shap[:,1,yy,xx]*143)
        
        
        # w=shap[:,0,yy,xx]
        # h=shap[:,1,yy,xx]
        
        anchor=np.zeros((cfg.TRAIN.BATCH_SIZE//cfg.TRAIN.NUM_GPU,size**2,4))

        anchor[:,:,0]=x+287//2
        anchor[:,:,1]=y+287//2
        anchor[:,:,2]=w
        anchor[:,:,3]=h


        return anchor
    def getcentercuda(self,mapp):

        def con(x):
            return x*143
        def dcon(x):
           x[t.where(x<=-1)]=-0.99
           x[t.where(x>=1)]=0.99
           return (t.log(1+x)-t.log(1-x))/2 
        
        size=mapp.size()[3]
        #location 
        x=t.Tensor(np.tile((16*(np.linspace(0,size-1,size))+63)-287//2,size).reshape(-1)).cuda()
        y=t.Tensor(np.tile((16*(np.linspace(0,size-1,size))+63).reshape(-1,1)-287//2,size).reshape(-1)).cuda()
        
        shap=dcon(mapp)*143
        
        xx=np.int16(np.tile(np.linspace(0,size-1,size),size).reshape(-1))
        yy=np.int16(np.tile(np.linspace(0,size-1,size).reshape(-1,1),size).reshape(-1))


        w=shap[:,0,yy,xx]+shap[:,1,yy,xx]
        h=shap[:,2,yy,xx]+shap[:,3,yy,xx]
        x=x-shap[:,0,yy,xx]+w/2+287//2
        y=y-shap[:,2,yy,xx]+h/2+287//2

        anchor=t.zeros((cfg.TRAIN.BATCH_SIZE//cfg.TRAIN.NUM_GPU,size**2,4)).cuda()

        anchor[:,:,0]=x-w/2
        anchor[:,:,1]=y-h/2
        anchor[:,:,2]=x+w/2
        anchor[:,:,3]=y+h/2
        return anchor
    
    def _convert_bbox(self, delta, anchor):
        delta = delta.contiguous().view(anchor.shape[0],4, -1)
        
        anchor=t.Tensor(anchor).cuda().float()
        locc=t.zeros_like(anchor).cuda()
        # x1y1x2y2 -->cxcywh
        locc[:,:,0] = delta[:,0, :]*143 +(anchor[:, :,0] -anchor[:,:, 2]/2 )
        locc[:,:,1] = delta[:,2, :]*143 +(anchor[:, :,1] -anchor[:,:, 3]/2 )
        locc[:,:,2] = (anchor[:, :,0] +anchor[:,:, 2]/2 )-delta[:,1, :]*143
        locc[:,:,3] = (anchor[:, :,1] +anchor[:,:, 3]/2 )-delta[:,3, :]*143
        
        return locc
    
    # def _convert_bbox(self, delta, anchor):
    #     delta = delta.contiguous().view(anchor.shape[0],4, -1)
        
    #     anchor=t.Tensor(anchor).cuda().float()
    #     locc=t.zeros_like(anchor).cuda()
        
    #     locc[:,:,0] = delta[:,0, :] * anchor[:,:, 2] + anchor[:, :,0]
    #     locc[:,:,1] = delta[:,1, :] * anchor[:,:, 3] + anchor[:,:, 1]
    #     locc[:,:,2] = t.exp(delta[:,2, :]) * anchor[:,:, 2]
    #     locc[:,:,3] = t.exp(delta[:,3, :]) * anchor[:, :,3]
        
    #     loc=t.zeros_like(anchor).cuda()
    #     loc[:,:,0]=locc[:,:,0]-locc[:,:,2]/2
    #     loc[:,:,1]=locc[:,:,1]-locc[:,:,3]/2
    #     loc[:,:,2]=locc[:,:,0]+locc[:,:,2]/2
    #     loc[:,:,3]=locc[:,:,1]+locc[:,:,3]/2
        
    #     return loc
    def _convert_score(self, score):
        score = score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)
        score = F.softmax(score, dim=1).data[:, 1].cpu().numpy()
        return score
    
    def transform(self,center):
        xx, yy, ww, hh = center[:,:,0], center[:,:,1], center[:,:,2], center[:,:,3]
        x1 = (xx - ww * 0.5).view(center[:,:,0].size())
        y1 = (yy - hh * 0.5).view(center[:,:,0].size())
        x2 = (xx + ww * 0.5).view(center[:,:,0].size())
        y2 = (yy + hh * 0.5).view(center[:,:,0].size())
        return  t.cat((x1.unsqueeze(-1),y1.unsqueeze(-1),x2.unsqueeze(-1),y2.unsqueeze(-1)),-1)

    def forward(self,data):
        """ only used in training
        """
                
        template = data['template'].cuda()
        search =data['search'].cuda()
        bbox=data['bbox'].cuda()
        labelcls2=data['label_cls2'].cuda()
        labelxff=data['labelxff'].cuda()
        labelcls3=data['labelcls3'].cuda()
        weightxff=data['weightxff'].cuda()
        

        
        zf = self.backbone(template)
        xf = self.backbone(search)
        loc,cls2,cls3=self.grader(xf,zf)
        #att=self.att(xf,zf)
       
        cls2 = self.log_softmax(cls2) 

        
 
        cls_loss2 = select_cross_entropy_loss(cls2, labelcls2)
        cls_loss3 = self.cls3loss(cls3, labelcls3)  
        
        pre_bbox=self.getcentercuda(loc) 
        bbo=self.getcentercuda(labelxff) 
        
        loc_loss=cfg.TRAIN.w3*self.IOULOSS(pre_bbox,bbo,weightxff) #cfg.TRAIN.w2*l1loss(loc,labelxff,weightxff)#+cfg.TRAIN.w3*self.IOULOSS(pre_bbox,bbo,weightxff)   #self.IOULOSS(pre_bbox,bbo,weightxff) #
       
        cls_loss=cfg.TRAIN.w4*cls_loss2+cfg.TRAIN.w5*cls_loss3
        # ind_loss=cfg.TRAIN.w6*self.INDLOSS()
        
        # score,best_idx=self.get_bestidx(cls2,cls3)
        
#        ind_loss=cfg.TRAIN.w6*self.INDLOSS(cls2,cls3,labelcls2,labelcls3)
       # ind_loss=0


        # label_cls,label_loc,label_loc_weight\
        #     =self.fin2.get(anchors,bbox,anchormap.size()[3])


        

        # cls1,cls2,cls3,loc=self.new(anchorfeature,original,att)
        
        # cls1 = self.log_softmax(cls1)  
        # cls2 = self.log_softmax(cls2) 

        
        # cls_loss1 = select_cross_entropy_loss(cls1, label_cls)
        # cls_loss2 = select_cross_entropy_loss(cls2, labelcls2)
        # cls_loss3 = self.cls3loss(cls3, labelcls3)  

        # cls_loss= (cfg.TRAIN.w3*cls_loss3 + cfg.TRAIN.w1*cls_loss1 + cfg.TRAIN.w2*cls_loss2)/(cfg.TRAIN.w3+cfg.TRAIN.w2+cfg.TRAIN.w1)

        # loc_loss1 = weight_l1_loss(loc, label_loc, label_loc_weight) 


        # pre_bbox=self._convert_bbox(loc,anchors)
        # label_bbox=self._convert_bbox(label_loc,anchors)
        
        # loc_loss2=self.IOULOSS(pre_bbox,label_bbox,label_loc_weight)

        # loc_loss=(cfg.TRAIN.w4*loc_loss1+cfg.TRAIN.w5*loc_loss2)/(cfg.TRAIN.w4+cfg.TRAIN.w5)
        
        # shapeloss=l1loss(anchormap,labelxff,weightxff) 
        
        

        outputs = {}
        outputs['total_loss'] =\
            cfg.TRAIN.LOC_WEIGHT*loc_loss\
                +cfg.TRAIN.CLS_WEIGHT*cls_loss\
                  #   +cfg.TRAIN.IND_WEIGHT*ind_loss\
                 #   +cfg.TRAIN.SHAPE_WEIGHT*shapeloss
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss
        #outputs['ind_loss'] = ind_loss
        #outputs['shapeloss'] = shapeloss
                                                    #2 4 1  都用loss2

        return outputs
