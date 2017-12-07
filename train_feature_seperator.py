#seperates the features in the datasets 
import csv
import sys


f_obj=open("simple_train_dataset.csv","rb")
f2=csv.reader(f_obj)

header=next(f2)
# next(header)
file=header[0].split('\t')


# print file	

# for row in f:
# 	print row
# 	break	
c=0
with open("simple_train_dataset.csv","rb") as f,open("train dataset/"+file[0]+".csv","wb") as f1,open("train dataset/"+file[1]+".csv","wb") as f2,\
open("train dataset/"+file[2]+".csv","wb") as f3,\
open("train dataset/"+file[3]+".csv","wb") as f4,open("train dataset/"+file[1]+"_"+file[3]+".csv","wb") as f5, open("train dataset/"+file[2]+"_"+ file[1]+ ".csv","wb") as f6, open("train dataset/"+file[2]+"_without_hashtags"+".csv","wb") as f7, open("dev dataset/"+file[0]+".csv","wb") as f21,open("dev dataset/"+file[1]+".csv","wb") as f22,\
open("dev dataset/"+file[2]+".csv","wb") as f23,\
open("dev dataset/"+file[3]+".csv","wb") as f24,open("dev dataset/"+file[1]+"_"+file[3]+".csv","wb") as f25, open("dev dataset/"+file[2]+"_"+ file[1]+ ".csv","wb") as f26, open("dev dataset/"+file[2]+"_without_hashtags"+".csv","wb") as f27:
	next(f)

	for row in f:
		
		row=row.split('\t')


		i=0
		temp=""

		if c%8==7:
			for col in row:
			
				if(i==0):
					f21.write(col+"\n")
				elif(i==1):
					temp=col
					f22.write(col+"\n")
					f25.write(col+"\t")
				elif(i==2):
					# if(c==0):
					# 	print col.split()
					# 	c=1
					l=col.split()
					l2=[]
					for word in l:
						if(word[0]!='#'):
							l2.append(word)
					tweet=' '.join(word for word in l2)
					f27.write(tweet+'\n')		
					f23.write(col+"\n")
					f26.write(col+" "+temp+"\n")
				elif(i==3):
					f24.write(col)
					f25.write(col)		
				i+=1
		else:
			for col in row:
			
				if(i==0):
					f1.write(col+"\n")
				elif(i==1):
					temp=col
					f2.write(col+"\n")
					f5.write(col+"\t")
				elif(i==2):
					# if(c==0):
					# 	print col.split()
					# 	c=1
					l=col.split()
					l2=[]
					for word in l:
						if(word[0]!='#'):
							l2.append(word)
					tweet=' '.join(word for word in l2)
					f7.write(tweet+'\n')		
					f3.write(col+"\n")
					f6.write(col+" "+temp+"\n")
				elif(i==3):
					f4.write(col)
					f5.write(col)		
				i+=1			

		c+=1
f_obj.close()