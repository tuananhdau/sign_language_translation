import threading


class SentenceBuilder:

    def __init__(self):

        self.text = ""

        self.lock = threading.Lock()


    # =====================================================
    # STATIC - THÊM CHỮ CÁI
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
    # DYNAMIC - THÊM TỪ / CỤM TỪ
    # =====================================================

    def add_phrase(self, phrase):

        if not phrase:
            return self.get_text()

        phrase = str(phrase).strip()

        if not phrase:
            return self.get_text()

        with self.lock:

            # Nếu phía trước đã có chữ
            # thì tự thêm khoảng trắng
            if (
                self.text
                and
                not self.text.endswith(" ")
            ):

                self.text += " "

            self.text += phrase

            return self.text


    # =====================================================
    # KHOẢNG TRẮNG
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
    # XÓA KÝ TỰ CUỐI
    # =====================================================

    def delete_last(self):

        with self.lock:

            if self.text:

                self.text = self.text[:-1]

            return self.text


    # =====================================================
    # XÓA TẤT CẢ
    # =====================================================

    def clear(self):

        with self.lock:

            self.text = ""

            return self.text


    # =====================================================
    # GET TEXT
    # =====================================================

    def get_text(self):

        with self.lock:

            return self.text