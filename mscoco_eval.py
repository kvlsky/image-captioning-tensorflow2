import sys
sys.path.append('coco-captions')
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap


def eval_scores(res_file=None):
    annotation_file = 'coco-captions/annotations/captions_val2014.json'
    result_file = res_file

    # create coco object and cocoRes object
    coco = COCO(annotation_file)
    cocoRes = coco.loadRes(result_file)

    # create cocoEval object by taking coco and cocoRes
    cocoEval = COCOEvalCap(coco, cocoRes)

    # evaluate on a subset of images by setting
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    # please remove this line when evaluating the full validation set

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    cocoEval.evaluate()

    scores = ['CIDEr', 'ROUGE_L', 'METEOR', 'Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4']
    scores_dict = dict(filter(lambda elem: elem[0] in scores, cocoEval.eval.items()))
    return scores_dict


if __name__ == "__main__":
    print(f'scores dict: {eval_scores("coco-captions/results/captions_val2014_fakecap_results.json")}')
