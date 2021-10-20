class MakhlukHidup():

    def __init__(self):
        self.name = 'notset'

    def setName(self, name = ''):
        self.name = name
    
    def getName(self):
        return self.name

class Burung(MakhlukHidup):
    def __init__(self, name = ''):
        self.name = name
    
    def setName(self, name = ''):
        self.name = name + ' macho'


burung = MakhlukHidup()
print(burung.getName())
burung.setName('Gagak')
print(burung.getName())

burungMacho = Burung('Burung Gagak')
print(burungMacho.getName())
burungMacho.setName('Bukan Gagak')
print(burungMacho.getName())
