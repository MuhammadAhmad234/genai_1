from pydantic import BaseModel

class Student(BaseModel):
    name: str


student = {'name': "John"}
student_obj = Student(**student)
print(student_obj)