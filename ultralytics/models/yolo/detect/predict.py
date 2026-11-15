                                      

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import ops


class DetectionPredictor(BasePredictor):
    

    def postprocess(self, preds, img, orig_imgs):
        
        preds = ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            classes=self.args.classes,
        )

        if not isinstance(orig_imgs, list):                                               
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        ir_result = []
        for i, pred in enumerate(preds):
                                     
                    
            orig_img = orig_imgs[i][..., 3:]                                 
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            img_path = self.batch[0][i]
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
                   
            ir_path = img_path.split('images')
            ir_path = str(ir_path[0] + 'image' + ir_path[1])
            if orig_imgs[i].shape[-1] >= 4:
                ir_img = orig_imgs[i][..., :3]
                ir_result.append(Results(ir_img, path=ir_path, names=self.model.names, boxes=pred))
        return results, ir_result
