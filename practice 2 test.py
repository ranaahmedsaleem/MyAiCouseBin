studentage=int(19)
studentheight=float(5.8)
sum = studentage+studentheight
studentname= "ahmed saleem"
print("sum of age and height =",sum)
print("type of sum of age and height =",type(sum))

subtract=studentage-studentheight
print("subtract of age and height =",subtract)
print("type of subtract of age and height =",type(subtract))

if (studentage>studentheight):
    print("age is greater than height")
elif (studentage<studentheight):
    print("height is greater than age")
else:
    print("age and height are equal")
    

#string operations
info=str("My name is Ahmed. My age is 19. My height is 5.8")
namestr=info[10:16:]
print(namestr)
agestr=info[27:30:]
print(agestr)
heightstr=info[44:48:]
print(heightstr)
#reversing
print("lenghth=",len(info))
rnamestr=info[16:10:-1]
print(namestr)
ragestr=info[30:27:-1]
print(agestr)
heightstr=info[48:44:-1]
print(heightstr)

#creating a list
studentlist=[studentname,studentage,studentheight]
print("student list =",studentlist)
print("student name from list =",studentlist[0])

studentlist.append("455")
print("student list after append =",studentlist)
studentlist.insert(1,"200")
print("student list after insert =",studentlist)
studentlist.remove(studentheight)
print("student list after remove =",studentlist)

#creating a tuple
studenttuple=(studentname,studentage,studentheight)
print("student tuple =",studenttuple)
print("student name from tuple =",studenttuple[0])

#createing a set
studentset={"ahmed",19,5.8}
print("student set =",studentset)
print("student name from set =",studentset)
studentset.add("7635e3")
print("student set after add =",studentset)
studentset.remove(19)

#createing a dictionary
studentdict={"name":"ahmed saleem","age":19,"height":5.8}
print("student dictionary =",studentdict)
print("student name from dictionary =",studentdict["name"])
studentdict["age"]=20