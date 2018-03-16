from sklearn.metrics import f1_score,precision_score,recall_score,precision_recall_curve
from itkutilities import get_itk_array
import numpy as np
import sys

def call_f1scores(grdtruth,prediction,masks):

    prec,rec,thres = precision_recall_curve(grdtruth.flatten(),prediction.flatten())

    f1 = (2*prec*rec)/(prec+rec)

    armax = np.argmax(f1)
    max_thres = thres[armax]
    img1 = np.asarray(prediction > max_thres,dtype=int)
    print('threshold :',max_thres)
    print('dice:',f1[armax])
    # print 'accuracy:',1-(np.sum(np.asarray(img1!=grdtruth,dtype=int)))


if __name__ == '__main__':

    scores = []#np.zeros(20)
    labels = []#np.zeros(20)
    preds = []#np.zeros(20)
    masks = []#np.zeros(20)

    for num in range(int(sys.argv[1]), int(sys.argv[2])):

        d = "%01d" % num

	labels.append(get_itk_array('labels_srxray/'+d+'.nii.gz'))
	preds.append(get_itk_array('confmaps_matthias/'+d+'.mhd'))

	# pred = np.asarray(pred/128,dtype='int32')
	# pred = np.flipud(np.fliplr(pred))

	# print np.unique(label),np.unique(pred)

	# print np.mean(label==pred)


	#print f1_score(label.flatten(),pred.flatten())
	#print precision_score(label.flatten(),pred.flatten())
	#print recall_score(label.flatten(),pred.flatten())
	#f1 = f1_score(label.flatten(),pred.flatten(),sample_weight=mask.flatten())
	#precision = precision_score(label.flatten(),pred.flatten(),sample_weight=mask.flatten())
	#recall = recall_score(label.flatten(),pred.flatten(),sample_weight=mask.flatten())

	#print num, " : ", f1
	#scores[num-1] = f1
    labels = np.array(labels)
    preds = np.array(preds)
    call_f1scores(labels,preds,masks)
 
#print scores
