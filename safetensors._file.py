import os, sys, json

class SafeTensorsException(Exception):
    def __init__(self, msg:str):
        self.msg=msg
        super().__init__(msg)

    @staticmethod
    def invalid_file(filename:str, whatiswrong:str):
        s=f"{filename} is not a valid .safetensors file: {whatiswrong}"
        return SafeTensorsException(msg=s)

    def __str__(self):
        return self.msg

class SafetensorsFile:
    def __init__(self):
        self.f = None
        self.hdrbuf = None
        self.header = None
        self.error = 0
        
    def __del__(self):
        self.close_file()
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.close_file()
        
    def _check_duplicate_keys(self):
        def parse_object_pairs(pairs):
            return [k for k, _ in pairs]
        
        keys = json.loads(self.hdrbuf, object_pairs_hook=parse_object_pairs)
        d = {}
        for k in keys:
            if k in d: d[k] = d[k] + 1
            else: d[k] = 1
            
            has_error = False
            for k, v in d.items():
                if v > 1:
                    print(f"Key {k} used {v} times in header", file = sys.stderr)
                    has_error = True
                    
            if has_error:
                raise SafeTensorsException.invalid_file(self.filename, "Duplicate keys in header")
        
    def close_file(self):
        if self.f is not None:
            self.f.close()
            self.f = None
            self.filename = ""
    
    @staticmethod
    def open_file(filename:str, quiet=False, parse_header=True):
        s = SafetensorsFile()
        s.open(filename, quiet, parse_header)
        return s

    def open(self, filename:str, quiet=False, parse_header=True) -> int:
        
        # Test File - header_zero.safetensors
        safetensor = os.stat(filename)
        if safetensor.st_size < 8:
            raise SafeTensorsException.invalid_file(filename, "Length less than 8 bytes")
        
        file = open(filename, "rb")
        
        # Header check
        header_size = file.read(8)
        if len(header_size) != 8:
            raise SafeTensorsException.invalid_file(filename, f"Read only {len(header_size)} bytes at start of file.")
        header_len = int.from_bytes(header_size, 'little', signed=False)
        
        # Test file - header_size_too_big.safetensors
        if (8 + header_len > safetensor.st_size):
            raise SafeTensorsException.invalid_file(filename, "Header extends past end of file.")
        
        if quiet == False:
            print(f"{filename}: length = {safetensor.st_size}, header length = {header_len}")
        
        hdrbuf = file.read(header_len)
        if len(hdrbuf) != header_len:
            raise SafeTensorsException.invalid_file(filename, f"Header size is {header_len}, but read {len(hdrbuf)} bytes") 
        
        self.filename = filename
        self.f = file
        self.hdrbuf = hdrbuf
        self.error = 0
        self.header_len = header_len
        
        if parse_header == True:
            self._check_duplicate_keys()
            self.header = json.loads(self.hdrbuf)
        return 0
            
        

if __name__=="__main__":
    
    safetensor_path = "./models/sd/sd_xl_base_1.0.safetensors"
    st_file = SafetensorsFile()
    st_file.open(safetensor_path,quiet=True)
    print(st_file.header["__metadata__"])

    
        