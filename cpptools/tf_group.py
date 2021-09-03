# coding=utf-8
import tensorflow as tf
import wml_tfutils as wmlt
import tfop
from object_detection2.standard_names import *
import wmodule
from object_detection2.datadef import *
from object_detection2.config.config import global_cfg
from object_detection2.modeling.build import HEAD_OUTPUTS
import object_detection2.odtools as odtl
import object_detection2.keypoints as kp
import basic_tftools as btf
import wnnlayer as wnnl

class TFGroup():
    def __init__(self):
        pass

    @wmlt.add_name_scope
    def inference(self, tag_k,loc_k,val_k,det,tags):
        ans = self.match(tag_k,loc_k,val_k)
        #ans = tf.Print(ans,[ans],summarize=1000)
        ans = self.adjust(ans,det=det)
        #ans = tf.Print(ans,[ans],summarize=1000)
        #ans = tfop.hr_net_refine(ans,det=det,tag=tags)
        #ans = tf.Print(ans,[ans],summarize=1000)

        scores = ans[...,2]
        org_scores = ans[...,2]
        scores = tf.reduce_mean(scores,axis=-1,keepdims=False)
        x,y = tf.unstack(ans[...,:2],axis=-1)
        mask = tf.greater(scores,0.2)
        size = wmlt.combined_static_and_dynamic_shape(x)[1]
        x,output_lens = wmlt.batch_boolean_mask(x,mask,size=size,return_length=True)
        y = wmlt.batch_boolean_mask(y,mask,size=size)
        pscores = wmlt.batch_boolean_mask(org_scores,mask,size=size)
        scores = wmlt.batch_boolean_mask(scores,mask,size=size)
        keypoints = tf.stack([x,y,pscores],axis=-1)

        return keypoints


    @btf.add_name_scope
    def adjust(self,ans,det):
        locs = ans[...,:2]
        values = ans[...,2]
        x,y = tf.unstack(locs,axis=-1)
        org_x,org_y = x,y
        xx = tf.cast(x,tf.int32)
        yy = tf.cast(y,tf.int32)
        B,H,W,num_keypoints = btf.combined_static_and_dynamic_shape(det)
        det = tf.transpose(det,[0,3,1,2])
        det = tf.reshape(det,[B*num_keypoints,H*W])
        yy_p = tf.minimum(yy+1,H-1)
        yy_n = tf.maximum(yy-1,0)
        xx_p = tf.minimum(xx+1,W-1)
        xx_n = tf.maximum(xx-1,0)

        def get_values(_xx,_yy):
            B,N,KN = btf.combined_static_and_dynamic_shape(_xx)
            _xx = tf.transpose(_xx,[0,2,1])
            _yy = tf.transpose(_yy,[0,2,1])
            _xx = tf.reshape(_xx,[B*KN,N])
            _yy = tf.reshape(_yy,[B*KN,N])
            index = _xx+_yy*W
            vals = tf.batch_gather(det,index)
            vals = tf.reshape(vals,[B,KN,N])
            vals = tf.transpose(vals,[0,2,1])
            return vals

        y_p = y+0.25
        y_n = y-0.25
        x_p = x+0.25
        x_n = x-0.25

        y = tf.where(get_values(xx,yy_p)>get_values(xx,yy_n),y_p,y_n)
        x = tf.where(get_values(xx_p,yy)>get_values(xx_n,yy),x_p,x_n)

        x = x+0.5
        y = y+0.5

        x = tf.where(values>0,x,org_x)
        y = tf.where(values>0,y,org_y)

        loc = tf.stack([x,y],axis=-1)

        B,N,KP,C = btf.combined_static_and_dynamic_shape(ans)
        _,data = tf.split(ans,[2,C-2],axis=-1)

        return tf.concat([loc,data],axis=-1)

    def match(self,tag_k,loc_k,val_k):
        res = tf.map_fn(lambda x:tfop.match_by_tag(x[0],x[1],x[2],
                                         detection_threshold=0.1,
                                         tag_threshold=1.0,
                                         use_detection_val=True),
                        elems=(tag_k,loc_k,val_k),dtype=tf.float32,back_prop=False)
        return res