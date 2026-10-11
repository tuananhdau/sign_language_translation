// =========================================================
// GLOBAL STATE
// =========================================================

let currentMode = "static";

let isChangingMode = false;


// =========================================================
// SET MODE
// =========================================================

async function setMode(
    mode
) {

    if (isChangingMode) {

        return;
    }


    if (
        mode !== "static"
        &&
        mode !== "dynamic"
    ) {

        return;
    }


    isChangingMode = true;


    try {

        const response = await fetch(
            "/set_mode",
            {

                method: "POST",

                headers: {

                    "Content-Type":
                        "application/json"

                },

                body: JSON.stringify({

                    mode: mode

                })

            }
        );


        const data = await response.json();


        if (!data.success) {

            console.log(
                "Khong the chuyen mode."
            );

            return;
        }


        currentMode = mode;


        updateModeUI(
            mode
        );


        resetStatusUI();


    } catch (error) {

        console.error(
            "Loi set mode:",
            error
        );

    } finally {

        isChangingMode = false;

    }

}


// =========================================================
// UPDATE MODE UI
// =========================================================

function updateModeUI(
    mode
) {

    const staticBtn =
        document.getElementById(
            "staticBtn"
        );


    const dynamicBtn =
        document.getElementById(
            "dynamicBtn"
        );


    const currentModeText =
        document.getElementById(
            "currentMode"
        );


    const description =
        document.getElementById(
            "modeDescription"
        );


    const modelName =
        document.getElementById(
            "modelName"
        );


    const sequenceContainer =
        document.getElementById(
            "sequenceContainer"
        );


    if (
        mode === "dynamic"
    ) {

        staticBtn.classList.remove(
            "active"
        );


        dynamicBtn.classList.add(
            "active"
        );


        currentModeText.innerText =
            "Ký hiệu động";


        modelName.innerText =
            "LSTM";


        description.innerText =
            "LSTM nhận diện chuyển động từ chuỗi 30 frame video.";


        sequenceContainer.style.display =
            "block";

    } else {

        dynamicBtn.classList.remove(
            "active"
        );


        staticBtn.classList.add(
            "active"
        );


        currentModeText.innerText =
            "Ký hiệu tĩnh";


        modelName.innerText =
            "Random Forest";


        description.innerText =
            "Random Forest nhận diện ký hiệu tĩnh từ landmark bàn tay.";


        sequenceContainer.style.display =
            "none";

    }

}


// =========================================================
// RESET STATUS UI
// =========================================================

function resetStatusUI() {

    document.getElementById(
        "prediction"
    ).innerText = "-";


    document.getElementById(
        "confidence"
    ).innerText = "0%";


    document.getElementById(
        "confirmed"
    ).innerText = "-";


    document.getElementById(
        "sequence"
    ).innerText = "0/30";


    document.getElementById(
        "sequenceBar"
    ).style.width = "0%";


    setRecognitionStatus(
        false
    );

}


// =========================================================
// UPDATE STATUS
// =========================================================

async function updateStatus() {

    try {

        const response = await fetch(
            "/status",
            {

                cache: "no-store"

            }
        );


        if (!response.ok) {

            return;
        }


        const data =
            await response.json();


        // =============================================
        // MODE
        // =============================================

        if (
            data.mode
            &&
            data.mode !== currentMode
        ) {

            currentMode =
                data.mode;


            updateModeUI(
                currentMode
            );

        }


        // =============================================
        // PREDICTION
        // =============================================

        document.getElementById(
            "prediction"
        ).innerText =
            data.prediction || "-";


        // =============================================
        // CONFIDENCE
        // =============================================

        let confidence = Number(
            data.confidence || 0
        );


        document.getElementById(
            "confidence"
        ).innerText =
            confidence.toFixed(1)
            + "%";


        // =============================================
        // CONFIRMED
        // =============================================

        document.getElementById(
            "confirmed"
        ).innerText =
            data.confirmed || "-";


        // =============================================
        // TEXT
        // =============================================

        updateResultText(
            data.text || ""
        );


        // =============================================
        // DYNAMIC SEQUENCE
        // =============================================

        if (
            data.mode === "dynamic"
        ) {

            const sequence =
                Number(
                    data.sequence || 0
                );


            const maxSequence =
                Number(
                    data.sequence_length
                    || 30
                );


            document.getElementById(
                "sequence"
            ).innerText =
                sequence
                + "/"
                + maxSequence;


            let percent = 0;


            if (
                maxSequence > 0
            ) {

                percent =
                    (
                        sequence
                        /
                        maxSequence
                    )
                    * 100;

            }


            if (
                percent > 100
            ) {

                percent = 100;

            }


            document.getElementById(
                "sequenceBar"
            ).style.width =
                percent + "%";

        }


        // =============================================
        // WAIT STATUS
        // =============================================

        setRecognitionStatus(
            Boolean(
                data.waiting
            )
        );


    } catch (error) {

        console.error(
            "Loi lay status:",
            error
        );

    }

}


// =========================================================
// UPDATE RESULT TEXT
// =========================================================

function updateResultText(
    text
) {

    const resultText =
        document.getElementById(
            "resultText"
        );


    if (
        text
        &&
        text.trim() !== ""
    ) {

        resultText.innerText =
            text;


        resultText.classList.remove(
            "placeholder"
        );

    } else {

        resultText.innerText =
            "Kết quả sẽ xuất hiện tại đây...";


        resultText.classList.add(
            "placeholder"
        );

    }

}


// =========================================================
// RECOGNITION STATUS
// =========================================================

function setRecognitionStatus(
    waiting
) {

    const status =
        document.getElementById(
            "recognitionStatus"
        );


    if (waiting) {

        status.innerText =
            "Đã nhận diện - hãy bỏ tay ra để tiếp tục";


        status.classList.add(
            "waiting"
        );

    } else {

        if (
            currentMode === "dynamic"
        ) {

            status.innerText =
                "Đang thu chuỗi chuyển động...";

        } else {

            status.innerText =
                "Sẵn sàng nhận diện";

        }


        status.classList.remove(
            "waiting"
        );

    }

}


// =========================================================
// ADD SPACE
// =========================================================

async function addSpace() {

    try {

        const response = await fetch(
            "/add_space",
            {

                method: "POST"

            }
        );


        const data =
            await response.json();


        if (data.success) {

            updateResultText(
                data.text || ""
            );

        }


    } catch (error) {

        console.error(
            "Loi add space:",
            error
        );

    }

}


// =========================================================
// DELETE LAST
// =========================================================

async function deleteLast() {

    try {

        const response = await fetch(
            "/delete_last",
            {

                method: "POST"

            }
        );


        const data =
            await response.json();


        if (data.success) {

            updateResultText(
                data.text || ""
            );

        }


    } catch (error) {

        console.error(
            "Loi delete:",
            error
        );

    }

}


// =========================================================
// CLEAR TEXT
// =========================================================

async function clearText() {

    try {

        const response = await fetch(
            "/clear",
            {

                method: "POST"

            }
        );


        const data =
            await response.json();


        if (data.success) {

            updateResultText(
                ""
            );


            resetStatusUI();

        }


    } catch (error) {

        console.error(
            "Loi clear:",
            error
        );

    }

}


// =========================================================
// TEXT TO SPEECH
// =========================================================

function speakText() {

    const resultElement =
        document.getElementById(
            "resultText"
        );


    let text =
        resultElement.innerText;


    if (
        resultElement.classList.contains(
            "placeholder"
        )
    ) {

        return;
    }


    if (
        !text
        ||
        text.trim() === ""
    ) {

        return;
    }


    if (
        !(
            "speechSynthesis"
            in window
        )
    ) {

        alert(
            "Trình duyệt không hỗ trợ đọc văn bản."
        );

        return;
    }


    // Dừng giọng cũ
    window.speechSynthesis.cancel();


    const speech =
        new SpeechSynthesisUtterance(
            text
        );


    speech.lang =
        "vi-VN";


    speech.rate =
        1.0;


    speech.pitch =
        1.0;


    speech.volume =
        1.0;


    window.speechSynthesis.speak(
        speech
    );

}


// =========================================================
// KEYBOARD SHORTCUT
// =========================================================

document.addEventListener(
    "keydown",
    function (
        event
    ) {

        // Ctrl + Backspace
        // Xóa toàn bộ kết quả

        if (
            event.ctrlKey
            &&
            event.key === "Backspace"
        ) {

            event.preventDefault();


            clearText();

        }

    }
);


// =========================================================
// INITIALIZE
// =========================================================

document.addEventListener(
    "DOMContentLoaded",
    function () {

        updateModeUI(
            "static"
        );


        updateStatus();


        // Cập nhật khoảng 3 lần / giây
        setInterval(
            updateStatus,
            300
        );

    }
);