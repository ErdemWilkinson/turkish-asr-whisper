# Kaliteli Türkçe ASR kayıt protokolü

Common Voice çeşitlilik sağlar; gerçek cihaz performansı için bunun yanında aynı
mikrofonla toplanmış, denetlenmiş kayıtlar gerekir. Hedef v1: en az 12 konuşmacı,
her biri 80 farklı kısa cümleyi iki kez okur. Bu, yaklaşık 1.920 kayıt eder.

Her kayıt mono, 16 kHz, 16-bit PCM WAV olmalı. Konuşmacı mikrofondan 15–30 cm
uzakta ve doğal hızda konuşur. Her cümlenin önünde ve sonunda 200 ms sessizlik
bırakılır. Aynı cümlenin ikinci tekrarı farklı odada veya düşük seviyeli arka
plan gürültüsüyle alınır.

Kayıtlar şu yapıda tutulur; bu klasör Git tarafından yok sayılır:

```text
voice/data/raw_v2/<speaker_id>/<session>/<prompt_id>.wav
voice/data/raw_v2/<speaker_id>/<session>/transcript.tsv
```

`transcript.tsv` başlığı: `prompt_id<TAB>transcript`. Cihaz için ayrı test
konuşmacıları ayırın: 8 kişi eğitim, 2 kişi doğrulama, 2 kişi test. Aynı kişinin
kayıtları hiçbir zaman birden fazla bölüme girmemelidir.

Reddetme ölçütleri: kesilmiş kelime, belirgin klipleme, yanlış/eksik metin,
çift konuşma, 0,8 sn’den kısa veya 12 sn’den uzun kayıt. Fısıltı hedefi yeniden
eklenirse aynı protokole her konuşmacı için ek bir `whisper` oturumu eklenir.
