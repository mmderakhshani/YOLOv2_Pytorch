from pycocotools.coco import COCO
import cv2
import numpy as np
import pdb

dataDir='../..'
dataType='train2014'
annFile='%s/annotations/instances_%s.json'%(dataDir,dataType)

coco=COCO(annFile)

# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
# nms=[cat['name'] for i,cat in enumerate(cats)]
for i,cat in enumerate(cats):
    print(i+1, cat['name'])
# print('COCO categories: \n', ' '.join(nms))

nms = set([cat['supercategory'] for cat in cats])
print('\nCOCO supercategories: \n', ' '.join(nms))

# get all images containing given categories, select one at random
# catIds = coco.getCatIds(catNms=['person','dog','skateboard']);
catIds = coco.getCatIds(catNms=['person']);
imgIds = coco.getImgIds(catIds=catIds );
img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]

I = cv2.imread('../../train2014/%s'%(img['file_name']))
annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
anns = coco.loadAnns(annIds)

pdb.set_trace()
a = anns[0]
x1 = int(a['bbox'][0])
y1 = int(a['bbox'][1])
x2 = int(a['bbox'][0]) + int(a['bbox'][2])
y2 = int(a['bbox'][1]) + int(a['bbox'][3])
cv2.rectangle(I, (x1,y1),(x2,y2),(0,255,0),2) 
cv2.imshow("MainWindows", I)
cv2.waitKey(0)
cv2.destroyAllWindows() 