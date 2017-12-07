from sklearn.metrics import f1_score

y_true=[]
y_pred=[]

with open("Prediction.csv","rb") as f1:
	for row in f1:
		col=row.split('\t')
		y_true.append(col[1].strip())
		y_pred.append(col[0].strip())
		

f=f1_score(y_true,y_pred,average=None)
f1=(f[0]+f[1])/2
print f1
