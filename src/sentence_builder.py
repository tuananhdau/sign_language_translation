import threading


class SentenceBuilder:

    def __init__(self):

        self.text = ""
        self.lock = threading.Lock()


    # =====================================================
    # CHỮ CÁI: A B C...
    # =====================================================

    def add_letter(self, letter):

        if not letter:
            return self.get_text()

        letter = str(letter).strip().upper()

        if not letter:
            return self.get_text()

        with self.lock:

            self.text += letter

            return self.text


    # =====================================================
    # TỪ / CỤM TỪ: Xin chào, Xin lỗi...
    # =====================================================

    def add_phrase(self, phrase):

        if not phrase:
            return self.get_text()

        phrase = str(phrase).strip()

        if not phrase:
            return self.get_text()

        with self.lock:

            if (
                self.text
                and
                not self.text.endswith(" ")
            ):
                self.text += " "

            self.text += phrase

            return self.text


    # =====================================================
    # SPACE
    # =====================================================

    def add_space(self):

        with self.lock:

            if (
                self.text
                and
                not self.text.endswith(" ")
            ):
                self.text += " "

            return self.text


    # =====================================================
    # DELETE
    # =====================================================

    def delete_last(self):

        with self.lock:

            if self.text:
                self.text = self.text[:-1]

            return self.text


    # =====================================================
    # CLEAR
    # =====================================================

    def clear(self):

        with self.lock:

            self.text = ""

            return self.text


    # =====================================================
    # GET
    # =====================================================

    def get_text(self):

        with self.lock:
            return self.text