import xport


with open('DIQ.XPT', 'rb') as f:
    for row in xport.Reader(f):
        print(row)