from .politenessr import Politenessr

pr = None

def predict(text):
    global pr
    if pr is None:
        try:
            pr = Politenessr()
        except Exception as e:
            print("Failed to initialize Politenessr")
            print(e)

    return pr.predict(text)