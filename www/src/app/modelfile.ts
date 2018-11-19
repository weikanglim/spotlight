export class ModelFile {
    name: string;
    file: File;

    toFormData() : FormData {
        var formData = new FormData();
        formData.append("name", this.name);
        formData.append("file", this.file);

        return formData;
    }
}