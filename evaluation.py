import numpy as np
import my_accumarray as accum
def getPR(scores, labels, numgt):
  #sort in decreasing order of scores
  i = np.argsort(-scores)
  labels = labels[i]
  scores = scores[i]
  tp = np.cumsum(labels)
  fp = np.cumsum(1-labels)
  prec = tp/(tp+fp)
  rec = tp/numgt

  mrec = rec.copy()
  mprec = prec.copy() 
  mrec = mrec.reshape(-1,1)
  mprec = mprec.reshape(-1,1)
  mrec = np.vstack((0.0, mrec, 1.0))
  mprec = np.vstack((0.0, mprec, 0.0))
  #precision should be non increasing
  for i in xrange(mprec.size-1,0,-1):
     mprec[i-1] = max(mprec[i], mprec[i-1])

  #find points where the recall is not the same
  i = np.where(mrec[1:]!=mrec[:-1])[0]
  #sum
  ap = np.sum((mrec[i+1]-mrec[i])*mprec[i+1])
  return ap, prec, rec


def compute_overlap_sprep(sp, reg2sp, inst):
  #compute intersection between all instances and all sp
  intsp = accum.my_accumarray((inst.reshape(-1), sp.reshape(-1)-1),1, (np.max(inst)+1,np.max(sp)))
  
  #compute the total intersection
  R=reg2sp.astype(np.float32, copy=True)
  inters = np.dot(intsp, R)
  #to compute the union, compute region areas
  spareas = accum.my_accumarray(sp.reshape(-1)-1,1,np.max(sp))
  totalareas = np.dot(spareas.reshape(1,-1),R)

  #compute instance areas
  instareas = accum.my_accumarray(inst.reshape(-1),1,np.max(inst)+1)
  #union is sum of areas - inters
  uni = instareas.reshape(-1,1)+totalareas.reshape(1,-1)-inters
  #union that is 0 should be set to 1 for safety
  uni[uni==0]=1.
  ov = inters/uni
  #the fist instance is bg so remove it
  ov = ov[1:,:]
  return ov

def assign_dets_to_gt(scr, ov, thresh):
  if ov.size==0:
    return np.zeros(scr.shape)

  #assign everything
  assigned = np.argmax(ov,0)
  bestov = np.max(ov,0)
  #now go down ranked list
  covered = [False]*ov.shape[0]
  labels = np.zeros(scr.size)
  idx = np.argsort(-scr)
  for i in idx:
    if not covered[assigned[i]] and bestov[i]>=thresh:
      covered[assigned[i]]=True
      labels[i] = 1
  return labels


def generalized_det_eval_simple(scores, ov, gt, categid, thresh):
  #we will count and gt first
  numdets = 0
  numgt = 0  
  for k in range(len(scores)):
    categids = gt[k]
    numdets += scores[k].size
    numgt += np.sum(categids==categid)
    
  #init
  allscores = np.zeros(numdets)
  alllabels = np.zeros(numdets)
  count = 0
  for k in range(len(scores)):
    #if no dets here continue
    if scores[k].size==0:
      continue
    #otherwise, assign
    idx = np.where(gt[k]==categid)[0]
    labelstmp = assign_dets_to_gt(scores[k].reshape(-1),ov[k][idx,:].reshape(-1, scores[k].size), thresh)
    allscores[count:count+scores[k].size] = scores[k].reshape(-1)
    alllabels[count:count+scores[k].size] = labelstmp.reshape(-1)
    count +=scores[k].size

  #compute PR
  [ap, prec, rec] = getPR(allscores, alllabels, numgt)  
  return ap, prec, rec



