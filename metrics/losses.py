from keras import backend as K

### NOTE: Assuming Backend is Tensorflow ###

'''
Format - (1, W, H, D)
'''

import tensorflow as tf

### Regression Loss Parameters ###

lambda_rpn_regr = 1.0
lambda_rpn_class = 1.0

epsilon = 1e-4


### RPN Loss Functions ###

def rpn_loss_regr(n_anchors):
    def rpn_loss_regr_fixed_num(y_true, y_pred):
        ### y_true = [y_rpn_overlap, y_regr] ###
        ### Therefore, y_true[:,:,:,4*anchors] are the required labels. ###
        '''
        Huber's Loss - 

        H{a}(y_true, y_pred):
        x = |y_true - y_pred|
        
        1. (0.5) * x^2 if x <= a
        2. a * (x - 0.5 * a) otherwise.
        '''

        x = y_true[:, :, :, 4 * n_anchors:] - y_pred
        x_abs = K.abs(x)
        
        ### Only Overlapping Anchors are Valid Labels ###
        
        valid_labels = y_true[:, :, :, :4 * n_anchors]
        n_valid_labels = K.sum(epsilon + valid_labels)
        
        x_abs_less_than_a = K.cast(K.less_equal(x_abs, lambda_rpn_regr), dtype = tf.float32)        
        
        ### Loss ###

        loss_1 = K.sum(valid_labels * x_abs_less_than_a * (x * x * 0.5))
        loss_2 = K.sum(valid_labels * lambda_rpn_regr * (1 - x_abs_less_than_a) * (x_abs - 0.5 * lambda_rpn_regr))

        ### Normalisation of Loss ###
        
        huber_loss = (loss_1 + loss_2) / n_valid_labels
        
        return huber_loss
    
    return rpn_loss_regr_fixed_num


def rpn_loss_cls(n_anchors):
    def rpn_loss_cls_fixed_num(y_true, y_pred):        
        ### Valid Boxes ###
        
        valid_labels = y_true[:, :, :, :n_anchors]
        n_valid_labels = K.sum(epsilon + valid_labels)

        ### Loss ###

        bce_loss = lambda_rpn_class * (K.sum(valid_labels * K.binary_crossentropy(y_true[:,:,:,n_anchors:], y_pred)))
        
        ### Normalisation ###

        bce_loss /= n_valid_labels
        return bce_loss
    
    return rpn_loss_cls_fixed_num 