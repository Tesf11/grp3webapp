from app.hf_loader import HFAdapter
m = HFAdapter("app/models/prodcat_model", max_length=128)
print(m.predict(["Canon EOS 200D DSLR Camera", "Adidas Copa soccer boots"]))
