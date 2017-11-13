import csv
#try:
with open("test.csv", "w", encoding='shift_jis') as csvfile:
    writer=csv.writer(csvfile,lineterminator='\n')
    writer.writerow(['id','family','first'])
    writer.writerow([1,'2','3'])
    writer.writerow([2,'4','5'])
    writer.writerow([3,'6','7'])
#exceptFileNotFoundError as e:
    #print(e)
#exceptcsv.Error as e:
    #print(e)
