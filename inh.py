class ANIMAL:
    def __init__(self, name):  
        self.name = name

    def speak(self):
        print(self.name)

class Dog(ANIMAL):  
    def speak(self):
        super().speak()
        print(self.name)

animal = ANIMAL("CAT")
dog = Dog("ZEBRA")  
dog.speak()
