ขั้นตอนการ run 
dea_with_metric.py

ทุกระดับภาษามี 3333 prompts

1.  copy file ระดับภาษา .jsonl จาก data\enron\context_thai_five_hiarchy\.jsonl มาที่ data\enron\.jsonl

2.  เปลี่ยนชื่อไฟล์ใน attacks\DataExtraction\enron.py 
    def __int__
        self.context = load_jsonl(os.path.join(data_path, "ระดับภาษาที่รัน.jsonl"))

3.  run dea_with_metric.py 
    3.1     เลือกชนิดการโจมตี 
    3.2     จำนวน samples ที่ใช้โจมตี
    3.3     model ที่จะโจมตี

4.  เมื่อ run ครบจำนวน sample ที่ input 
    ไฟล์ผลการโจมตีอยู่ที่ output/dea/dea_output_เลขจำนวนไฟล์
    เปลี่ยนชื่อไฟล์เป็น dea_output_ระดับของภาษา_promptเริ่ม_promptสุดท้าย
    และย้ายไฟล์ไปที่ dea_result/ชนิดการโจมตี/th/modelที่ใช้/ระดับภาษา/dea_output_ระดับของภาษา_model_promptเริ่ม_promptสุดท้าย

5.  ก่อนเริ่ม run อีกครั้งกรุณาลบ prompt ในไฟล์ data/enron/ระดับของภาษา.jsonl ตามจำนวนที่ใช้โจมตี
6.  และย้าย prompt ที่ใช้แล้วไปไว้ใน data\enron\completed\ระดับภาษา\ระดับภาษา_promptเริ่ม_promptสุดท้าย.jsonl
