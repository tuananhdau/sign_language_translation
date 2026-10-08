import threading


class SentenceBuilder:

    def __init__(
        self
    ):

        self.text = ""

        self.lock = (
            threading.Lock()
        )


    # =====================================================
    # ADD CHARACTER
    # =====================================================

    def add_character(
        self,
        character
    ):

        if not character:

            return self.get_text()


        with self.lock:

            self.text += str(
                character
            )


            return self.text


    # =====================================================
    # ADD WORD
    # =====================================================

    def add_word(
        self,
        word
    ):

        if not word:

            return self.get_text()


        word = str(
            word
        ).strip()


        if not word:

            return self.get_text()


        with self.lock:

            if (
                self.text

                and

                not self.text.endswith(
                    " "
                )
            ):

                self.text += " "


            self.text += word


            return self.text


    # =====================================================
    # SPACE
    # =====================================================

    def add_space(
        self
    ):

        with self.lock:

            if (
                self.text

                and

                not self.text.endswith(
                    " "
                )
            ):

                self.text += " "


            return self.text


    # =====================================================
    # DELETE CHARACTER
    # =====================================================

    def delete_last(
        self
    ):

        with self.lock:

            if self.text:

                self.text = (
                    self.text[:-1]
                )


            return self.text


    # =====================================================
    # DELETE WORD
    # =====================================================

    def delete_last_word(
        self
    ):

        with self.lock:

            cleaned = (
                self.text.rstrip()
            )


            if not cleaned:

                self.text = ""

                return self.text


            words = (
                cleaned.split()
            )


            words = (
                words[:-1]
            )


            self.text = (
                " ".join(
                    words
                )
            )


            return self.text


    # =====================================================
    # CLEAR
    # =====================================================

    def clear(
        self
    ):

        with self.lock:

            self.text = ""

            return self.text


    # =====================================================
    # GET TEXT
    # =====================================================

    def get_text(
        self
    ):

        with self.lock:

            return self.text


    # =====================================================
    # CHECK EMPTY
    # =====================================================

    def is_empty(
        self
    ):

        with self.lock:

            return (
                len(
                    self.text.strip()
                )
                == 0
            )