import { ClassLabelResult } from "./class-label-result";

export class TrainingResult {
    modelName: string;
    accuracy: number;
    classificationMatrix: ClassLabelResult[];
    modelUri: string;
}

