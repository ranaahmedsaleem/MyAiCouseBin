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
print("type of student list =",type(studentlist))
print("length of student list =",len(studentlist))
print("student name from list =",studentlist[0])
print("student age from list =",studentlist[1])
print("student height from list =",studentlist[2])
print("type of student name from list =",type(studentlist[0]))
print("type of student age from list =",type(studentlist[1]))