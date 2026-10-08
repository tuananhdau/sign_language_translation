try:

    import pyttsx3

except ImportError:

    pyttsx3 = None


class TextToSpeech:

    def __init__(
        self,
        rate=150,
        volume=1.0
    ):

        self.engine = None


        if pyttsx3 is None:

            print(
                "Chua cai pyttsx3."
            )

            print(
                "Chay: pip install pyttsx3"
            )

            return


        try:

            self.engine = (
                pyttsx3.init()
            )


            self.engine.setProperty(
                "rate",
                rate
            )


            self.engine.setProperty(
                "volume",
                volume
            )


        except Exception as error:

            print(
                "Khong khoi tao duoc TTS:"
            )

            print(
                error
            )

            self.engine = None


    # =====================================================
    # SPEAK
    # =====================================================

    def speak(
        self,
        text
    ):

        if self.engine is None:

            return False


        if not text:

            return False


        text = str(
            text
        ).strip()


        if not text:

            return False


        try:

            self.engine.say(
                text
            )

            self.engine.runAndWait()

            return True


        except Exception as error:

            print(
                "Loi Text To Speech:"
            )

            print(
                error
            )

            return False


    # =====================================================
    # STOP
    # =====================================================

    def stop(
        self
    ):

        if self.engine is None:

            return


        try:

            self.engine.stop()

        except Exception:

            pass